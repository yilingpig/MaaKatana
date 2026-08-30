import sys

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

import my_action
import my_reco
import quantity_reco
import recognition_click
import building_router
import dynamic_swipe
import dynamic_bidirectional_swipe
import reward_reco
import condition_router
import exchange_router
import exchange_slider
import nonogram_solver
import color_nonogram_solver
import color_nonogram_test
import color_nonogram_paint_test


def main():
    Toolkit.init_option("./")

    if len(sys.argv) < 2:
        print("Usage: python main.py <socket_id>")
        print("socket_id is provided by AgentIdentifier.")
        sys.exit(1)

    socket_id = sys.argv[-1]

    AgentServer.start_up(socket_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
