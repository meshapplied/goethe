"""goethe serve | list | doctor."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence

from goethe import __version__, connect, doctor, engine, fit, hw, registry, resolve, store

PROG = "goethe"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description="Run Goethe models locally.")
    p.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    serve = sub.add_parser("serve", help="install if needed and start the server",
                           description="Start an OpenAI and Anthropic compatible server. "
                                       "Everything after -- goes to the engine unchanged.",
                           usage="goethe serve [options] [model] [-- engine args]")
    serve.add_argument("model", nargs="?", help="model id, default the first one in `goethe list`")
    serve.add_argument("--profile", choices=resolve.PROFILES, default="agent",
                       help="agent (default), chat or long; --explain shows what each one sets")
    serve.add_argument("--host", help="default 127.0.0.1; any other address makes goethe use an API key")
    serve.add_argument("--port", help="default 8080")
    serve.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                       help=f"override a setting ({', '.join(resolve.VALIDATORS)}); repeatable")
    serve.add_argument("--explain", action="store_true",
                       help="show every setting, where it came from and why, then exit")
    serve.add_argument("--dry-run", action="store_true", help="show the plan and the commands, run nothing")

    ls = sub.add_parser("list", help="models and whether this GPU can run them")
    ls.add_argument("--json", action="store_true")

    cn = sub.add_parser("connect", help="run a coding agent against the server, for one command")
    cn.add_argument("harness", help="claude, codex, aider, opencode, qwen, goose, kilo or mini-swe-agent")
    cn.add_argument("--port", type=int, default=8080)
    cn.add_argument("--print", dest="show", action="store_true", help="print the command, run nothing")

    dr = sub.add_parser("doctor", help="check this machine, change nothing")
    dr.add_argument("--json", action="store_true")
    dr.add_argument("--port", type=int, default=8080)
    return p


def _flags(args: argparse.Namespace) -> dict[str, str]:
    flags = {}
    for item in args.sets:
        key, sep, value = item.partition("=")
        if not sep:
            raise resolve.ResolveError(f"--set expects KEY=VALUE, got {item!r}")
        flags[key] = value
    for key in ("host", "port"):
        if getattr(args, key):
            flags[key] = getattr(args, key)
    return flags


def _explain(plan: resolve.Plan) -> str:
    g = plan.gpu
    lines = [f"gpu      {g.name}, {g.total_gib:.1f} GiB, sm_{g.compute_cap.replace('.', '')}, driver {g.driver}: "
             f"{plan.tier.name} ({plan.tier.reason})",
             f"model    {plan.model.id} ({plan.model.title}), engine {plan.model.engine}",
             f"profile  {plan.profile}"]
    for key, s in plan.settings.items():
        shown = "<hidden>" if key == "api_key" else s.value
        lines.append(f"{key:<13}{shown!s:<12} {s.source}" + (f": {s.why}" if s.why else ""))
    lines += [f"warning  {w}" for w in plan.warnings]
    return "\n".join(lines)


def cmd_serve(args: argparse.Namespace) -> int:
    models = registry.load()
    model = registry.find(models, args.model) if args.model else next(iter(models.values()))
    gpus, err = hw.probe()
    if not gpus:
        print(f"error: {err}. Run `{PROG} doctor` for details.", file=sys.stderr)
        return 1
    config = resolve.load_config(store.config_path())
    plan = resolve.resolve(model, gpus, args.profile, flags=_flags(args), config=config, extra_args=args.engine_args)
    inst = engine.locate(model)
    if args.explain or args.dry_run:
        print(_explain(plan))
    if args.dry_run:
        for cmd in engine.commands(inst):
            print("run     ", shlex.join(cmd))
        print("exec    ", shlex.join(resolve.server_argv(plan, inst.binary, inst.model_file, redact=True)))
        print(f"fit      context and lanes probed against the engine, cached in {fit.tune_path()}")
    if args.explain or args.dry_run:
        return 0
    host, port = str(plan.value("host")), int(plan.value("port"))
    if engine.healthy(host, port):
        print(f"already serving on http://{host}:{port}")
        return 0
    if engine.port_in_use(host, port):
        print(f"error: port {port} is in use but nothing answers on /health yet; "
              "a server may still be starting, or pick another port with --port", file=sys.stderr)
        return 1
    engine.ensure(inst)
    argv = resolve.server_argv(plan, inst.binary, inst.model_file)
    result = _fitted(plan, argv, inst.model_file)
    if (result.context, result.lanes) != (plan.value("context"), plan.value("lanes")):
        print(f"fitted to {result.lanes} lanes and {result.context} context "
              f"({result.kv_tokens} KV tokens, {result.attempts} boots)")
    plan.settings["context"].value, plan.settings["lanes"].value = result.context, result.lanes
    argv = resolve.server_argv(plan, inst.binary, inst.model_file)
    if plan.settings.get("api_key", resolve.Setting("", "")).source == "generated":
        print(f"API key: {store.state_dir() / 'api_key'}")
    print(f"serving {plan.model.served_name} on http://{host}:{port} "
          f"({plan.value('lanes')} lanes, {plan.value('context')} context, {plan.profile})", flush=True)
    os.execv(argv[0], argv)


def _fitted(plan: resolve.Plan, argv: list[str], model_file) -> fit.Fit:
    """The cached fit when it still applies, otherwise a fresh probe saved for next time."""
    key, path, free = fit.cache_key(plan, model_file), fit.tune_path(), fit.free_mib()
    if (cached := fit.load(path, key, free)):
        return fit.Fit(cached["context"], cached["lanes"], cached["kv_tokens"], 0)
    result = fit.fit(plan, argv, path.with_suffix(".log"))
    fit.save(path, key, result, free)
    return result


def cmd_connect(args: argparse.Namespace) -> int:
    model = next(iter(registry.load().values()))
    try:
        cached = json.loads(fit.tune_path().read_text()) if fit.tune_path().exists() else {}
    except (OSError, json.JSONDecodeError):
        cached = {}
    context = cached.get("context", model.profiles["agent"].context) if isinstance(cached, dict) else \
        model.profiles["agent"].context
    env, argv = connect.command(args.harness, model.served_name, args.port, context)
    argv += args.engine_args
    if args.harness == "kilo":
        root = store.state_dir() / "connect-kilo"
        env["XDG_CONFIG_HOME"] = str(root)
        body = connect.kilo_config(connect.BASE.format(port=args.port), model.served_name)
    else:
        root, body = None, None
    if args.show:
        if body:
            print(body)
        for key, value in env.items():
            print(f"{key}={value}")
        print(shlex.join(argv))
        return 0
    if not engine.healthy("127.0.0.1", args.port):
        print(f"error: nothing is serving on port {args.port}. Run `goethe serve` first.", file=sys.stderr)
        return 1
    missing = connect.conforms(args.port)
    if missing:
        print(f"error: the server is missing {', '.join(missing)}", file=sys.stderr)
        return 1
    if body:
        dest = root / "kilo"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "kilo.jsonc").write_text(body)
    binary = shutil.which(argv[0])
    if binary is None:
        print(f"error: {argv[0]} is not installed", file=sys.stderr)
        return 1
    os.execve(binary, argv, {**os.environ, **env})


def cmd_list(args: argparse.Namespace) -> int:
    models = registry.load()
    gpus, err = hw.probe()
    rows = []
    for m in models.values():
        inst = engine.locate(m)
        if gpus:
            tier = resolve.tier_for(m, gpus[0])
            status, why = tier.name, tier.reason
        else:
            status, why = "no gpu", err
        rows.append({"id": m.id, "title": m.title, "engine": m.engine, "status": status, "why": why,
                     "installed": inst.has_binary and inst.has_model})
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['id']:<14} {r['title']:<20} {r['status']:<12} "
                  f"{'installed' if r['installed'] else 'not installed':<14} {r['why']}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    model = next(iter(registry.load().values()))
    inst = engine.locate(model)
    checks = doctor.run_checks(model, inst, registry.read_pin(inst.recipe / "NINFER_PIN"), args.port)
    if args.json:
        print(json.dumps(doctor.as_dicts(checks), indent=2))
    else:
        for c in checks:
            print(f"{c.status:<5} {c.name:<13} {c.detail}" + (f"\n{'':19}fix: {c.fix}" if c.fix else ""))
    return doctor.exit_code(checks)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    engine_args = []
    if "--" in argv:
        cut = argv.index("--")
        argv, engine_args = argv[:cut], argv[cut + 1:]
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if engine_args and args.command not in ("serve", "connect"):
        parser.error("arguments after -- only apply to serve and connect")
    args.engine_args = engine_args
    try:
        return {"serve": cmd_serve, "list": cmd_list, "doctor": cmd_doctor,
                "connect": cmd_connect}[args.command](args)
    except (resolve.ResolveError, registry.RegistryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"error: {shlex.join(map(str, e.cmd))} exited with {e.returncode}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
