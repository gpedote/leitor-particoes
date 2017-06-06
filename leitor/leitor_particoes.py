#!/usr/bin/python
# -*- coding: utf-8 -*-

from glob import glob
import os

import numpy as np
import pandas as pd

class Leitor(object):

    def __init__(self, separador="\t", verificar_tamanhos=True, verificar_ids=True):
        self.separador = separador
        self.verificar_tamanhos = verificar_tamanhos
        self.verificar_ids = verificar_ids

    def ler_particoes(self, dir_particoes, ext_particoes="*.clu"):
        dir_particoes = self.__adiciona_barra_ao_final_do_caminho(dir_particoes)
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
            if (primeira_particao):
                tam_particoes = tam_particao_atual
                ids_particoes = particao["id"]
            elif (self.verificar_tamanhos and tam_particoes != tam_particao_atual):
                raise RuntimeError("Partição de tamanho diferente das outras: ", file)
            elif (self.verificar_ids and self.__verificar_ids_iguais(ids_particoes, particao["id"])):
                raise RuntimeError("Partição com ids inconsistêntes: ", file)

            particoes.set_value(i, "nome_particao", arqvs_particoes[i])
            particoes.set_value(i, "particao", particao)

        return particoes

    def __ler_particao_ordenada(self, arq_particao):
        p = np.genfromtxt(arq_particao, dtype=None, delimiter=self.separador)

        # Se os dados forem homogêneos modifica o dtype para que a nomeação seja possível
        if (p.dtype.names is None):
            novo_dtype = map(lambda z : ('f{}'.format(z), p.dtype), range(0, p.shape[1]))
            p.dtype = np.dtype(novo_dtype)

        p.dtype.names = ("id", "label")
        p.sort(order="id")
        return p

    def __verificar_ids_iguais(self, ids1, ids2):
        return set(ids1) == set(ids2)

    def __obter_particoes(self, dir_particoes, ext_particoes):
        if not os.path.exists(dir_particoes):
            raise RuntimeError("Diretório com as partições não pode ser encontrado", dir_particoes)
        if not os.path.isdir(dir_particoes):
            raise RuntimeError("Caminho fornecido não é um diretório", dir_particoes)

        arqvs_particoes = glob(dir_particoes + ext_particoes)
        qtd_arqvs_particoes = len(arqvs_particoes)

        if qtd_arqvs_particoes == 0:
            raise RuntimeError("Não foram encontradas partições no diretorio: ", dir_particoes)

        return (qtd_arqvs_particoes, arqvs_particoes)


    def __adiciona_barra_ao_final_do_caminho(self, c):
        return c if c.endswith("/") else c + "/"
