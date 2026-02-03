import importlib


def get_eric_chat_mlx():
    try:
        mod = importlib.import_module("erictransformer")
        return getattr(mod, "EricChatMLX", None)
    except Exception:
        return False
