# Author Tacla, UTFPR
# First version  set/2025

import os
import pygame
import csv
import time
import math
import colorsys
from .physical_agent import PhysAgent
from .constants import VS


# Class Environment
class Env:
    def __init__(self, vict_folder, env_folder):
        # instance attributes
        self.vict_folder = vict_folder  # folder containing victims' data
        self.env_folder = env_folder    # folder containing env config data
        self.dic = {}          # configuration of grid and window
        self.agents = []       # list of running agents
        self.obst = None       # list of obstacles: ]0.0, VS.OBST_WALL] float
                               # representing the multiplying factor for the
                               # walk action for an agent to enter into a cell.
                               # explorer agent cannot access this attribute, it has to find!
        self.nb_of_victims = 0  # total number of victims
        self.victims = []       # positional: the coordinates of the victims [(x1,y1), ..., (xn, yn)]
        self.tri = []      # positional: the injury tri for each victim (label)
        self.sobr = []     # positional: the injury survival prob. for each victim (float value)
        self.sum_sobr = 0  # sum of all gravity values for peg and psg calculation
        self.signals = []  # positional: the vital signals of the victims [[i,s1,...,s5,g,l],...]
        self.found = [[]]  # positional: Physical agents that found each victim [[ag1] [ag2, ag3], ...] ag1 found vict 0, ag2 and 3, vict 1, ... 
        self.saved = [[]]  # positional: Physical agents that saved each victim
        self.__max_obst = 0  # max value for obstacle for coloring - to be calculated
        self.__min_obst = VS.OBST_WALL  # min value for obstacle for coloring - to be calculated

        # Read the environment config file
        self.__read_config()
        # print(self.dic)

        # Set up the obstacles - it's a list composed of GRID_WIDTH lists.
        # Each sublist is a column (y=0, 1, ...)
        # 1 means that there is no obstacle - it is a regular terrain
        self.obst = [[1 for y in range(self.dic["GRID_HEIGHT"])]
                     for x in range(self.dic["GRID_WIDTH"])]
        obst_file = os.path.join(self.env_folder, "env_obst.txt")
        self.__max_obst = 1

        with open(obst_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                x = int(row[0])
                y = int(row[1])
                # absolute multiplying factor representing the degree of
                # difficulty/facility for the agent to enter the cell
                # values ]0, 1[ means a descent; 1 = VS.OBST_NONE;
                # ]1, 100[ = ascent; 100 = VS.OBST_WALL
                obst = float(row[2])

                if obst > 100:
                    obst = VS.OBST_WALL   # wall
                elif obst <= 0:
                    obst = VS.OBST_NONE     # no obstacle

                if obst != VS.OBST_WALL and obst > self.__max_obst:
                    self.__max_obst = obst

                self.obst[x][y] = obst
                #  print(self.obst)

        print(f"ENV: max_obst = {self.__max_obst} min_obst={self.__min_obst}")

        #  Read and put the victims into the grid
        victims_file = os.path.join(self.env_folder, "env_victims.txt")

        with open(victims_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                x = int(row[0])
                y = int(row[1])
                self.victims.append((x, y))   # append tuples

        self.nb_of_victims = len(self.victims)

        # Load the vital signals of the victims
        vs_file = os.path.join(self.vict_folder, "data.csv")

        with open(vs_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Pula a primeira linha (cabeçalho)
            vid = 0
            for row in csvreader:
                vid = vid + 1   # victim id number
                idade = int(row[0])
                fc = int(row[1])    # freq cardiaca
                fr = int(row[2])    # freq respiratoria
                pas = int(row[3])   # pressao arterial sistolica
                spo2 = int(row[4])  # saturacao de oxigenio
                temp = float(row[5])  # temperatura corporal
                pr = int(row[6])    # pulso radial (0 ou 1)
                sg = int(row[7])    # sangramento (0 ou 1
                fx = int(row[8])    # fratura exposta (0 ou 1)
                queim = int(row[9])  # queimardura (niveis)
                gcs = int(row[10])   # Coma - Glasgow scale
                avpu = int(row[11])  # estado de consciencia
                tri = int(row[12])  # Triagem START: 0 GRN, 1 YEL, 2 RED, 3 BLK
                sobr = float(row[13])  # Prob. de sobrevivencia
                self.signals.append([vid, idade, fc, fr, pas, spo2, temp,
                                     pr, sg, fx, queim, gcs, avpu, tri, sobr])
                self.tri.append(tri)
                self.sobr.append(sobr)
                self.sum_sobr = self.sum_sobr + sobr

        if self.nb_of_victims > len(self.signals):
            print("ENV: number of victims of env_victims.txt greater than vital signals")
            print("ENV: end of execution")
            exit()

        if self.nb_of_victims < len(self.signals):
            print("ENV: nb of victims of env_victims.txt less than vital signals")
            print("ENV: Assuming nb of victims of env_victims.txt")

        # Set up found and saved victims' lists
        self.found = [[] for v in range(self.nb_of_victims)]
        self.saved = [[] for v in range(self.nb_of_victims)]

        # Stores all the agents have been in the cell
        self.visited = [[[] for y in range(self.dic["GRID_HEIGHT"])]
                        for x in range(self.dic["GRID_WIDTH"])]

    def __read_config(self):
        """ Read the size of the grid and window and
            loads into a dictionary """
        # Open config file
        size_file = os.path.join(self.env_folder, "env_config.txt")
        with open(size_file, "r") as file:
            # Read each line of the file
            for line in file:
                # Split the line into words
                words = line.split()

                # Get the keyword and value
                keyword = words[0]
                raw_value = words[1]

                # casts the value
                if keyword == "BASE":
                    value = [int(i) for i in raw_value.split(',')]
                elif keyword == "DELAY":
                    value = float(raw_value)
                else:
                    value = int(raw_value)

                self.dic[keyword] = value

    def add_agent(self, ag, state=VS.IDLE):
        """ This public method adds an agent to the simulator.
        It creates a representation for the agent in the 2D environment
        @param self: the environment object
        @param ag: an instance of Abstract Agent
        @param state: the state of the agent
        @return: an object that is the agent"""

        phy = PhysAgent(ag, self, self.dic["BASE"][0], self.dic["BASE"][1], state) 
        self.agents.append(phy)
        return phy

    def __draw(self):
        """ This private method draw the grid and its items """

        # Set cell width and height
        cell_w = self.dic["WINDOW_WIDTH"]/self.dic["GRID_WIDTH"]
        cell_h = self.dic["WINDOW_HEIGHT"]/self.dic["GRID_HEIGHT"]

        # Clear the screen
        self.screen.fill(VS.WHITE)

        # configuration for obstacles coloring
        # h,  s,   lc, ld:
        # 13, 100, 100, 65 red tonalities
        # 275,100, 100, 65 purple
        # 90, 35,  100, 40 green tonaliies
        #  0,  0,  100, 50 gray tonnalitites
        hue = 0                # Not relevant for grayscale; 0=Red, 120=green, 240=blue till 360
        saturation = 0         # 40 = Red  0 = Grayscale
        lightness_clear = 100  # 100 = White
        lightness_dark = 40    # 0 = Black

        # configuration for ploting the trace marks
        nb_of_ag = len(self.agents)
        # print(f"ENV: number of agents {nb_of_ag}")
        nb_of_rects = math.ceil(math.sqrt(nb_of_ag))
        mark_radius = min(cell_w/nb_of_rects, cell_h/nb_of_rects) / 2

        # Draw the grid
        for x in range(self.dic["GRID_WIDTH"]):
            for y in range(self.dic["GRID_HEIGHT"]):
                rect = pygame.Rect(x * cell_w, y * cell_h, cell_w, cell_h)
                pygame.draw.rect(self.screen, (230, 230, 230), rect, 1)

                if self.obst[x][y] == VS.OBST_WALL:
                    rgb_int = VS.BLACK
                else:
                    if self.obst[x][y] == VS.OBST_NONE:
                        rgb_int = VS.WHITE
                    else:
                        perc = self.obst[x][y]/self.__max_obst
                        lightness = (1 - perc) * lightness_clear + perc * lightness_dark

                        # convert HSL color to RGB
                        rgb_color = colorsys.hls_to_rgb(hue / 360.0, lightness / 100.0, saturation / 100.0)

                        # Convert RGB values to integers in the range [0, 255]
                        rgb_int = tuple(int(c * 255) for c in rgb_color)

                obst_rect = pygame.Rect(x * cell_w + 1, y * cell_h + 1, cell_w - 2, cell_h - 2)
                pygame.draw.rect(self.screen, rgb_int, obst_rect)

                # Trace: plot a dot for each agent who has visited a cell
                visitors = self.visited[x][y]
                v = 0

                if visitors:
                    for i in range(nb_of_rects):
                        for j in range(nb_of_rects):
                            if v < len(visitors):
                                trace_color = visitors[v].mind.TRACE_COLOR
                                xc = x * cell_w + mark_radius * (i+1)
                                yc = y * cell_h + mark_radius * (j+1)
                                pygame.draw.circle(self.screen, trace_color,
                                                   (xc, yc), 0.7*mark_radius)
                                v += 1

        # Draw a marker at the base
        rect = pygame.Rect(self.dic["BASE"][0] * cell_w,
                           self.dic["BASE"][1] * cell_h, cell_w, cell_h)
        pygame.draw.rect(self.screen, VS.CYAN, rect, 4)

        # Draw the victims
        v = 0
        for victim in self.victims:
            victim_rect = pygame.Rect(victim[0] * cell_w + 1,
                                      victim[1] * cell_h + 1,
                                      cell_w - 1, cell_h - 1)
            c = self.tri[v]
            pygame.draw.ellipse(self.screen, VS.VIC_COLOR_LIST[c], victim_rect)
            if self.saved[v] != []:
                pygame.draw.ellipse(self.screen, VS.BLUE, victim_rect, 3)
            elif self.found[v] != []:
                pygame.draw.ellipse(self.screen, VS.PINK, victim_rect, 3)
            v = v + 1

        # Draw the physical agents
        for phy in self.agents:
            if phy._state == VS.ACTIVE:
                # ag_rect = pygame.Rect(phy.x * cell_w, phy.y * cell_h, cell_w, cell_h)
                # pygame.draw.rect(self.screen, phy.mind.COLOR, ag_rect)
                p_x1 = phy.x * cell_w + 0.2 * cell_w
                p_x2 = phy.x * cell_w + cell_w/2
                p_x3 = phy.x * cell_w + 0.8 * cell_w
                p_y1 = phy.y * cell_h + cell_h/2
                p_y2 = phy.y * cell_h + 0.2 * cell_h
                p_y3 = phy.y * cell_h + 0.8 * cell_h

                triangle = [(p_x1, p_y1), (p_x2, p_y2),
                            (p_x3, p_y1), (p_x2, p_y3)]
                pygame.draw.polygon(self.screen, phy.mind.COLOR, triangle)
                # active_idle = True

        # Update the display
        pygame.display.update()

    def run(self):
        """ This public method is the engine of the simulator. It calls the
        deliberate method of each ACTIVE agent situated in the environment.
        Then, it updates the state of the agents and of the environment"""

        cycle = 0
        # Set up Pygame
        pygame.init()

        # Create the font object
        self.font = pygame.font.SysFont(None, 24)

        # Create the window
        self.screen = pygame.display.set_mode((self.dic["WINDOW_WIDTH"],
                                               self.dic["WINDOW_HEIGHT"]))

        # Draw the environment with items
        self.__draw()

        # Create the main loop
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # control whether there are active or idle agents
            active_or_idle = False

            # ask each agent to deliberate the next action
            for phy in self.agents:

                # Asks the agent to do the next action if it is ACTIVE
                if phy._state == VS.ACTIVE:
                    active_or_idle = True
                    more_actions_to_do = phy.mind.deliberate()

                    #  if cycle % 50 == 0:
                    #    print(f"ENV: cycle {cycle} {phy.mind.NAME} remaining: {phy.rtime}")

                    # Test if the agent exceeded the time limit
                    if phy._end_of_time():
                        phy._state = VS.DEAD
                        print("ENV: " + phy.mind.NAME + ": time limit reached, no batt, it is dead")
                    # agent do not have more actions to do
                    elif not more_actions_to_do:
                        if phy._at_base():
                            print("ENV: ag " + phy.mind.NAME + " succesfully terminated, it is at the base")
                            phy._state = VS.ENDED
                        else:
                            print("ENV: ag " + phy.mind.NAME + " is not at the base and asked for termination. Now, it's dead")
                            phy._state = VS.DEAD

                elif phy._state == VS.IDLE:
                    active_or_idle = True

            # Update the grid after the delay
            if self.dic["DELAY"] > 0:
                time.sleep(self.dic["DELAY"])

            self.__draw()

            cycle += 1

            # Show metrics when there is no more active or idle agents
            if not active_or_idle:
                print("ENV: no active or idle agent scheduled for execution... terminating")
                if self.dic["STATS_PER_AG"] == 1:
                    print("RESULTS PER AGENT")
                    self.print_results()

                if self.dic["STATS_ALL_AG"] == 1:
                    print("\n--------------")
                    self.print_acum_results()

                input("ENV: Tecle qualquer coisa para encerrar >>")
                running = False

        # Quit Pygame
        pygame.quit()

    def __print_victims(self, victims, type_str, sub, ident=3):
        """ Print either the found or the saved victims list
        @param victims: it is the list to be printed
        @param type_str: it is a string for composing the pring
        @param sub: it is a character representing the metric"""

        idents = ' ' * ident

        if len(victims) > 0:
            tri = []
            sobr = []
            tot_sobr = 0        # for peg or psg calculation
            for v in victims:
                tri.append(self.tri[v])
                sobr.append(self.sobr[v])
                tot_sobr = tot_sobr + self.sobr[v]

            print(f"\n{idents}{type_str} victims: (ID, Tri, Sobr)")
            for i in range(len(victims)):
                if (i % 4 == 0):
                    print("")
                print(f"{idents}({victims[i]:d}, {tri[i]:d}, {sobr[i]:.1f})", end=' ')

            print("\n")
            if self.tri.count(0) > 0:
                print(f"{idents}Green  {type_str}      (V{sub}0) = {tri.count(0):3d} out of {self.tri.count(0)} ({100*tri.count(0)/self.tri.count(0):.1f})%")
            if self.tri.count(1) > 0:
                print(f"{idents}Yellow {type_str}      (V{sub}1) = {tri.count(1):3d} out of {self.tri.count(1)} ({100*tri.count(1)/self.tri.count(1):.1f})%")
            if self.tri.count(2) > 0:
                print(f"{idents}Red    {type_str}      (V{sub}2) = {tri.count(2):3d} out of {self.tri.count(2)} ({100*tri.count(2)/self.tri.count(2):.1f})%")
            if self.tri.count(3) > 0:
                print(f"{idents}Black  {type_str}      (V{sub}3) = {tri.count(3):3d} out of {self.tri.count(3)} ({100*tri.count(3)/self.tri.count(3):.1f})%")
            print(f"{idents}--------------------------------------")
            print(f"{idents}Total of {type_str} victims     (V{sub})  = {len(tri):3d} ({100*float(len(tri)/self.nb_of_victims):.2f}%)")

            weighted = ((3*tri.count(0) + 3*tri.count(1) + 3*tri.count(2) + tri.count(3))/
                        (3*self.tri.count(0)+3*self.tri.count(1)+3*self.tri.count(2)+self.tri.count(3)))

            print(f"{idents}Weighted {type_str} victims per sobr (V{sub}g) = {weighted:.2f}\n")
            print(f"{idents}Sum of sobr of all {type_str} victims = {tot_sobr:.2f} of a total of {self.sum_sobr:.2f}")
            print(f"{idents}  % of sobr of all {type_str} victims = {tot_sobr/self.sum_sobr:.2f}")
            print(f"{idents}--------------------------------------")
            print(f"{idents}CSV of {type_str} victims")
            print(f"{idents}V{sub}0,V{sub}1,V{sub}2,V{sub}3,V{sub}g")
            print(f"{idents}{tri.count(0)},{tri.count(1)},{tri.count(2)},{tri.count(3)},{weighted}")
        else:
            print(f"{idents}No {type_str} victims")
            print(f"{idents}--------------------------------------")
            print(f"{idents}CSV of {type_str} victims")
            print(f"{idents}V{sub}1,V{sub}2,V{sub}3,V{sub}4,V{sub}g")
            print(f"{idents}0,0,0,0,0.0")

    def print_results(self):
        """ For each agent, print found victims and saved victims by sobr
        This is what actually happened in the environment. Observe that the
        beliefs of the agents may be different."""

        print("\n\n*** Final results per agent ***")
        for phy in self.agents:
            print(f"\n[ Agent {phy.mind.NAME} ]")
            if phy._state == VS.DEAD:
                print("This agent is dead, you should discard its results, but...")

            # Remaining time
            print("\n*** Consumed time ***")
            print(f"{phy.mind.TLIM - phy._rtime:.2f} of {phy.mind.TLIM:.2f}")

            # Found victims
            found = phy._get_found_victims()
            self.__print_victims(found, "found", "e", ident=5)

            # Saved victims
            saved = phy._get_saved_victims()
            self.__print_victims(saved, "saved", "s", ident=5)

    def print_acum_results(self):
        """ Print found victims and saved victims by severity for all agents.
        This is what actually happened in the environment"""

        print("\n\n*** ACUMULATED RESULTS - FOR ALL AGENTS ***\n")
        print(f" *** Numbers of Victims in the Environment ***")
        print(f"   Green  = {self.tri.count(0):3d}")
        print(f"   Yellow = {self.tri.count(1):3d}")
        print(f"   Red    = {self.tri.count(2):3d}")
        print(f"   Black  = {self.tri.count(3):3d}")
        print(f"   --------------------------------------")
        print(f"   Total of victims    (V)  = {self.nb_of_victims:3d}")
        print(f"   Sum of all gravities(SG) = {self.sum_sobr:.2f}")
        print(f"   --------------------------------------")
        print(f"   CSV of nb. total of victims")
        print(f"   G,Y,R,B,SSOBR")
        print(f"   {self.tri.count(0)},{self.tri.count(1)},{self.tri.count(2)} {self.tri.count(3)},{self.sum_sobr}")

        found = []
        for index, agents in enumerate(self.found, start=0):
            if agents:
                found.append(index)
        print("")
        print(" *** FOUND victims by all explorer agents ***")
        self.__print_victims(found, "found", "e", ident=5)

        saved = []
        for index, agents in enumerate(self.saved, start=0):
            if agents:
                saved.append(index)
        print("")
        print(" *** SAVED victims by all rescuer agents ***")
        self.__print_victims(saved, "saved", "s", ident=5)
        print("\n *** END OF STATS ***")
