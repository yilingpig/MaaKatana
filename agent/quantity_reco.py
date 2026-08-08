from maa.agent.agent_server import AgentServer

from quantity_router import ManufacturingQuantityRouter


AgentServer.custom_action("制造数量路由")(ManufacturingQuantityRouter)
