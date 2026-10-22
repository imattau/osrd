# Local OSRD demo authentication

This overlay configures a local-only mock identity named **OSRD Demo Planner**. The
mock provider has no password, so use it only on a machine-bound development stack.
It grants the OSRD `Admin` role for local fixture testing. Keep this mock identity on the
machine-bound development stack only.

Start or update the local stack with the overlay:

```sh
DOCKER_SOCKET="$XDG_RUNTIME_DIR/podman/podman.sock" \
podman compose \
  -f docker-compose.yml \
  -f docker/docker-compose.front.yml \
  -f ops-intelligence/docker-compose.yml \
  -f docker/docker-compose.demo-auth.yml \
  up -d --no-build
```

The front overlay starts the OSRD UI required by the gateway. The demo-auth overlay
seeds the demo account on Editoast startup and mounts the gateway mock provider
configuration. The gateway login endpoint issues the normal session cookie and binds
to `127.0.0.1:4000`; the planner API binds to `127.0.0.1:8095`. Open the OSRD front
end at `http://localhost:4000`. The OSRD login flow uses the default mock provider
(`POST /auth/login`) and authenticates as **OSRD Demo Planner** without a password.
The direct frontend, Editoast, database, broker, object store, and observability ports
are not published to the host by this overlay, so requests enter through the
authenticated gateway.

The account can access local fixture studies for testing. This identity and its mock
authentication are not evidence of operator authorization or planner participation.
