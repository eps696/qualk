"""qualk web app launcher.

    python app.py                              # http://localhost:8765, no login
    python app.py --simulate                   # keyless demo with synthetic sources (banner shown)
    python app.py --host 127.0.0.1             # accept connections from this machine only
    python app.py --token XYZ                  # optional: require an access token (or set QUALK_TOKEN)
    python app.py --ssl-certfile c.pem --ssl-keyfile k.pem   # TLS, or put a reverse proxy in front
"""

import argparse
import os
import sys

from qualk.security import is_loopback_host


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--runs', default='runs', help='folder holding one sub-folder per run')
    p.add_argument('--token', default=os.environ.get('QUALK_TOKEN', ''),
                   help='optional access token (or QUALK_TOKEN); without it there is no login')
    p.add_argument('--env', default='.env', help='where the Settings panel stores keys')
    p.add_argument('--simulate', action='store_true', help='synthetic embedder/web/LLM (no keys needed)')
    p.add_argument('--max-rounds', type=int, default=200)
    p.add_argument('--ssl-certfile', default=None)
    p.add_argument('--ssl-keyfile', default=None)
    a = p.parse_args(argv)

    token = a.token or None
    remote = not is_loopback_host(a.host)
    if remote and not token:
        print(f'Note: listening on {a.host} with no access token. Anyone who can reach this port can start, steer '
              'and delete runs and spend the configured API keys. Use --host 127.0.0.1 to keep it local, or --token.',
              file=sys.stderr)
    if remote and not a.ssl_certfile:
        print('Note: no TLS, so anything typed into Settings travels in clear text; use --ssl-certfile/--ssl-keyfile '
              'or a TLS reverse proxy when the server is not on a trusted network.', file=sys.stderr)

    import uvicorn
    from qualk.server import create_app
    app = create_app(a.runs, token=token, simulate=a.simulate, env_path=a.env, max_rounds=a.max_rounds)
    scheme = 'https' if a.ssl_certfile else 'http'
    print(f'qualk: {scheme}://{"localhost" if a.host in ("0.0.0.0", "::") else a.host}:{a.port}/'
          + ('   [SIMULATED sources]' if a.simulate else ''))
    uvicorn.run(app, host=a.host, port=a.port, log_level='warning',
                ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile)


if __name__ == '__main__':
    main()
