from .multi_agent_server import MultiAgentServer
from .client import AgentClientEnv
import weakref

def multi_agent_to_single_agent(
    multi_agent_env, asynchronous=True,
    return_server=False, ignore_multiple_reset=False
):
    server = MultiAgentServer(multi_agent_env, asynchronous=asynchronous)
    server.start()
    
    clients = [AgentClientEnv(weakref.proxy(multi_agent_env),
        server.csi, agentid, ignore_multiple_reset=ignore_multiple_reset)
            for agentid in range(server.num_agents)
    ]

    if return_server:
        return clients, server
    else:
        return clients
