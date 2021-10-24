from .async_server import MultiAgentServer
from .client import AgentClientEnv
import weakref

def multi_agent_to_single_agent(multi_agent_env, return_server=False):
    server = MultiAgentServer(multi_agent_env)
    server.start()
    
    clients = [AgentClientEnv(weakref.proxy(multi_agent_env),
        server.csi, agentid) for agentid in range(server.num_agents)]

    if return_server:
        return clients, server
    else:
        return clients
