#!/usr/bin/python
# -*- coding: utf-8 -*-

from glob import glob
import os

import numpy as np
import pandas as pd

class Leitor(object):

    def __init__(self, separador="\t"):
        self.separador = separador

    def ler_particoes(self, dir_particoes, ext_particoes="*.clu"):
        (qtd_arqvs_particoes, arqvs_particoes) = self.__obter_particoes(dir_particoes,
                ext_particoes)

        primeira_particao = True
        tam_particoes = 0
        ids_particoes = []
        indices_particoes = range(qtd_arqvs_particoes)
        particoes = pd.DataFrame([], index=indices_particoes, columns=["nome_particao", "particao"])

        for i in indices_particoes:
            particao = self.__ler_particao_ordenada(arqvs_particoes[i])

            # Garante que todas as partições tenham o mesmo tamanho e não estejam inconsistêntes
            tam_particao_atual = len(particao)
            if primeira_particao:
                tam_particoes = tam_particao_atual
                ids_particoes = particao["id"]
                primeira_particao = False
            else:
                if tam_particoes != tam_particao_atual:
                    raise RuntimeError("Partição de tamanho diferente das outras: ", file)
                if not self.__verificar_ids_iguais(ids_particoes, particao["id"]):
                    raise RuntimeError("Partição com ids inconsistêntes: ", file)

            particoes.set_value(i, "nome_particao", arqvs_particoes[i])
            particoes.set_value(i, "particao", particao)

        return particoes

    def __ler_particao_ordenada(self, arq_particao):
        p = np.genfromtxt(arq_particao, dtype=None, delimiter=self.separador, names="id, label")
        p.sort(order="id")
        return p

    def __verificar_ids_iguais(self, ids1, ids2):
        return np.array_equal(ids1, ids2)

    def __obter_particoes(self, dir_particoes, ext_particoes):
        if not os.path.exists(dir_particoes):
            raise RuntimeError("Diretório com as partições não pode ser encontrado", dir_particoes)
        if not os.path.isdir(dir_particoes):
            raise RuntimeError("Caminho fornecido não é um diretório", dir_particoes)

        dir_particoes = self.__adiciona_barra_ao_final_do_caminho(dir_particoes)

        arqvs_particoes = glob(dir_particoes + ext_particoes)
        qtd_arqvs_particoes = len(arqvs_particoes)

        if qtd_arqvs_particoes == 0:
            raise RuntimeError("Não foram encontradas partições no diretorio: ", dir_particoes)

        return (qtd_arqvs_particoes, arqvs_particoes)


    def __adiciona_barra_ao_final_do_caminho(self, c):
        return c if c.endswith("/") else c + "/"
