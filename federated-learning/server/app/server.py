import flwr as fl

def start_flower_server(session_id: str, rounds: int = 10, min_clients: int = 2):
    """Bridge for main.py to use the centralized manager."""
    from .server_manager_instance import fl_manager
    fl_manager.start_training(session_id, rounds, min_clients)
