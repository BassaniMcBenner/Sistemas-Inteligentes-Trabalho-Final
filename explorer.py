# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import random
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
import re
from collections import deque


class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def peek(self):
        if not self.is_empty():
            return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0


class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc):
        """Construtor do agente random on-line
        @param env: a reference to the environment
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc  # reference to the rescuer agent
        self.x = 0  # current x position relative to the origin 0
        self.y = 0  # current y position relative to the origin 0
        self.map = Map()  # create a map for representing the environment
        self.victims = {}  # a dictionary of found victims: (seq): ((x,y), [<vs>])
        # the key is a seq number of the victim,(x,y) the position, <vs> the list of vital signals

        self.visitados = set()  # posições visitadas
        self._frontier = []  # pilha de nós: [(pos, [vizinhos (ordem/cache)])]
        self.custo_volta = 1.05  # margem para custo de volta
        self.margem_seguranca = 1.575  # margem de seguranca

        # Descobre o número do agente a partir do nome (ex: EXPLORER_1 = idx = 0)
        agente_idx = 0
        m = re.search(r"_(\d+)$", self.NAME or "")
        if m:
            agente_idx = max(0, int(m.group(1)) - 1)

        # Sugestao sugerida pelo Gemini, com o intuito de diminuir sobreposicao
        direcao_prioridade = [
            [0, 1, 7, 2, 6, 3, 5, 4],  # 1: norte-leste
            [4, 5, 3, 6, 2, 7, 1, 0],  # 2: sul-oeste
            [2, 3, 1, 4, 0, 5, 7, 6],  # 3: leste-norte
        ]
        self.direcao_preferencia = deque(
            direcao_prioridade[agente_idx % len(direcao_prioridade)]
        )

        self.map.add((self.x, self.y), 1.0, VS.NO_VICTIM, self.check_walls_and_lim())
        self.visitados.add((self.x, self.y))

        # empilha nó raiz com vizinhos já ordenados
        self.stacka_no((self.x, self.y))

    def vizinh_livre(self, pos):
        """
        retorna os vizinhos que estão livres, retornando na hora das direcoes do agente
        """
        if pos == (self.x, self.y):
            # se estiver em cima, procura no ambiente
            parede = self.check_walls_and_lim()
        else:
            # tenta obter informacao pelo mapa
            parede = self._peek_walls_from(pos)

        vizin = []

        # verifica se as direcoes estao livre
        for direc in self.direcao_preferencia:
            if parede[direc] == VS.CLEAR:
                x, y = AbstAgent.AC_INCR[direc]
                nova_posi = (pos[0] + x, pos[1] + y)
                vizin.append(nova_posi)

        return vizin

    def _peek_walls_from(self, posi):
        """
        retorna a informacao sobre paredes e limites como se o agente
        estivesse nas coordenadas fornecidas
        """
        # se for a posição atual, consulta o ambiente
        if posi == (self.x, self.y):
            return self.check_walls_and_lim()

        # Caso contrário, pega as paredes salvas no mapa
        info_celula = self.map.map_data.get(posi)

        if info_celula is not None:
            _, _, paredes_salv = info_celula
            return list(paredes_salv)

        # Se a posicao nunca foi visitada, assume tudo bloqueado
        return [VS.BUMPED] * 8

    def stacka_no(self, pos):
        """empilha no com vizinhos que ainda não foram visitados, de acordo com as direcoes de prioridade."""
        vizinho = [n for n in self.vizinh_livre(pos) if n not in self.visitados]
        self._frontier.append([pos, vizinho])

    def custo_estimado_min(self, dx, dy):
        """
        retorna o custo estimado para dar um passo
        """
        # movimento na diagonal = custo diagonal
        if dx != 0 and dy != 0:
            return self.COST_DIAG
        # movimento em linha reta = custo linear
        return self.COST_LINE

    def continuar_explorando(self):
        """
        retorna se pode seguir explorando, verificando a bateria se tem o suficiente para voltar para base
        """
        # n de passos que o agente precisa dar para voltar
        passos_p_voltar = max(1, len(self._frontier))

        # estima o custo para voltar ate a base
        custo_estimado_volta = passos_p_voltar * self.COST_LINE * self.custo_volta

        # verifica se ainda tem tempo com folga
        temp_rest = self.get_rtime()
        return temp_rest >= custo_estimado_volta * self.margem_seguranca

    def movimentar(self, destino):
        """
        executa o movimento para a posicao informada, e se tiver vitima, realiza a leitura
        """

        # salva as coordenadas de destino e o seu deslocamento
        tx, ty = destino
        dx, dy = tx - self.x, ty - self.y

        # calcula o tempo gasto
        tempo_antes = self.get_rtime()
        resultado = self.walk(dx, dy)
        tempo_depois = self.get_rtime()

        # falha se tempo acabou ou tem obstaculo
        if resultado != VS.EXECUTED:
            return False

        self.x, self.y = tx, ty

        # calcula a dificuldade de se movimentar no terreno
        tempo_gasto = tempo_antes - tempo_depois
        custo_base = self.custo_estimado_min(dx, dy)
        dific = max(1.0, tempo_gasto / max(custo_base, 1e-9))

        # ve se tem vitima na posicao
        seq = self.check_for_victim()
        if seq != VS.NO_VICTIM:
            sinais_vitais = self.read_vital_signals()
            if sinais_vitais != VS.TIME_EXCEEDED:
                self.victims[seq] = ((self.x, self.y), sinais_vitais)

        # atualiza o mapa com as info
        self.map.add((self.x, self.y), dific, seq, self.check_walls_and_lim())

        return True

    def get_next_position(self):
        """
        escolhe o vizinho para explorar
        """
        if not self._frontier:
            return None
        _, vizinhos = self._frontier[-1]

        while vizinhos:
            proximo = vizinhos.pop(0)

            if proximo in self.visitados:
                continue

            # verifica se ha bateria para realizar a acao
            desloc_x = proximo[0] - self.x
            desloc_y = proximo[1] - self.y
            custo_ida = self.custo_estimado_min(desloc_x, desloc_y)

            passos_volta = max(1, len(self._frontier)) + 1
            custo_volta = passos_volta * self.COST_LINE * self.custo_volta

            tempo_necessario = (custo_ida + custo_volta) * self.margem_seguranca

            # se nao for possivel, ignora
            if self.get_rtime() < tempo_necessario:
                continue

            # retorna vizinho prox
            return proximo

        # caso nao ache, retorna nenhum
        return None

    def explore(self):
        """
        executa a exploracao
        """
        # pega o prox vizinh
        proximo = self.get_next_position()

        # caso nao haja vizinho valido, volta
        if proximo is None:
            # remove no da pilha se não houver vizinhos
            no_atual = self._frontier.pop() if self._frontier else None

            # se tem nós anteriores, volta p anterior
            if self._frontier:
                voltar_para, _ = self._frontier[-1]
                self.come_back(to_pos=voltar_para)
            return  #

        # caso haja, anda
        if self.movimentar(proximo):
            dx = proximo[0] - self.x
            dy = proximo[1] - self.y
            self.walk_stack.push((dx, dy))
            self.visitados.add(proximo)
            self.stacka_no(proximo)

    def come_back(self, to_pos=None):
        """
        agente volta.
        """
        if to_pos is None:
            # se tiver vazio, so volta
            if self.walk_stack.is_empty():
                return

            # tira da stack a ultima move
            dx, dy = self.walk_stack.pop()

            # inverte p voltar
            voltar_x, voltar_y = -dx, -dy

            # Executa o movimento de retorno
            resultado = self.walk(voltar_x, voltar_y)

            # realiza a volta
            if resultado == VS.EXECUTED:
                self.x += voltar_x
                self.y += voltar_y
            return

        # volta p no anterior
        desloc_x = to_pos[0] - self.x
        desloc_y = to_pos[1] - self.y
        resultado = self.walk(desloc_x, desloc_y)

        # anda se for bem sucedido
        if resultado == VS.EXECUTED:
            self.x, self.y = to_pos

    def deliberate(self) -> bool:
        """
        acao do agente
        """
        # se tiver fronteira e bateria no seguro, continua
        if self._frontier and self.continuar_explorando():
            self.explore()
            return True

        # vai voltando
        if self._frontier:
            if (self.x, self.y) != (0, 0):
                # se tiver um no só, vai direto p base. caso contrario, ele vai p no anterior
                volte = self._frontier[-2][0] if len(self._frontier) >= 2 else (0, 0)
                # realiza um passo de volta, se chegou no no anterior, remove ele da parte de cima da pilha
                self.come_back(to_pos=volte)
                if (self.x, self.y) == volte:
                    self._frontier.pop()
                return True

        # quando tiver na base, entrega os dados p socorristas
        if (self.x, self.y) == (0, 0):
            if callable(self.resc):
                self.resc(
                    self.NAME, self.map, self.victims
                )  # callback (nome, mapa, vítimas)
            else:
                self.resc.go_save_victims(
                    self.map, self.victims
                )  # objeto no estilo original

        return False
