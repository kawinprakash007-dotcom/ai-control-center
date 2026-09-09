import secrets
from fastapi import Header, HTTPException, Query, Request, WebSocket, status
from config.settings import get_settings


def verify_api_token(
    request: Request,
    authorization: str = Header(None, alias="Authorization"),
) -> str:
    """
    Constant-time comparison for API bearer/token authentication.
    Header format: 'Bearer <token>' or '<token>'
    """
    settings = getattr(request.app.state, "settings", None) or get_settings()
    expected_token = settings.api_auth_token

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.strip().split()
    provided_token = parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else parts[0]

    if not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or unauthorized API token",
        )
    return provided_token


def verify_ws_token(websocket: WebSocket, token: str = Query(None)) -> bool:
    """
    Verify WebSocket query token parameter: ws://host/path?token=<token>
    """
    settings = getattr(websocket.app.state, "settings", None) or get_settings()
    expected_token = settings.api_auth_token
    if not token or not secrets.compare_digest(token, expected_token):
        return False
    return True
