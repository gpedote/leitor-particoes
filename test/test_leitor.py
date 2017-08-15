#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import unittest

import numpy as np
import pandas as pd

from .context import Leitor

class TestParticoesRetornadas(unittest.TestCase):

    def setUp(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "iris-particoes",
                "iris-algPartitions-E")
        self.particoes = leitor.ler_particoes(caminho)

    def test_ler_todas_as_particoes(self):
        self.assertEqual(len(self.particoes), 37, "Deve ler todas as 37 partições")

    def test_garantir_nome_particao(self):
        particao = self.particoes.loc[0]
        self.assertIn("nome_particao", particao, "Deve conter o nome da partição")

    def test_garantir_dados_particao(self):
        particao = self.particoes.loc[0]
        self.assertIn("particao", particao, "Deve conter os dados da partição")
        dados_particao = particao["particao"].dtype.names
        self.assertIn("id", dados_particao, "Deve ler os ids da partição")
        self.assertIn("label", dados_particao, "Deve ler os labels da partição")

    def test_garantir_que_ids_estejam_ordenados(self):
        particao = self.particoes.loc[0]
        ids = particao["particao"]["id"]
        ids_ordenados = np.copy(ids)
        ids_ordenados.sort()
        self.assertTrue(np.array_equal(ids, ids_ordenados), "Os ids devem estar em ordem")

class TestIdsNumericos(unittest.TestCase):

    def setUp(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "iris-particoes",
                "iris-particoes-com-ids-numericos")
        self.particoes = leitor.ler_particoes(caminho)

    def test_ler_todas_as_particoes_com_ids_numericos(self):
        self.assertEqual(len(self.particoes), 3, "Deve ler todas as 3 partições")

    def test_garantir_que_ids_estejam_ordenados(self):
        particao = self.particoes.loc[0]
        ids = particao["particao"]["id"]
        ids_ordenados = np.copy(ids)
        ids_ordenados.sort()
        self.assertTrue(np.array_equal(ids, ids_ordenados), "Os ids devem estar em ordem")

class TestIdsInconsistentes(unittest.TestCase):

    def test_garantir_que_lanca_excessao_com_ids_inconsistentes(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "iris-particoes",
                "iris-particoes-com-erro")
        with self.assertRaises(RuntimeError):
            leitor.ler_particoes(caminho)

class TestTamanhosDiferentes(unittest.TestCase):

    def test_garantir_que_lanca_excessao_com_tamanhos_diferentes(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "iris-particoes",
                "iris-particoes-com-tamanhos-diferentes")
        with self.assertRaises(RuntimeError):
            leitor.ler_particoes(caminho)

class TestSemParticoes(unittest.TestCase):

    def test_garantir_que_lanca_excessao_caso_nao_ache_particoes(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "sem-particoes")
        with self.assertRaises(RuntimeError):
            leitor.ler_particoes(caminho)

    def test_garantir_que_lanca_excessao_caso_nao_ache_diretorio(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "diretorio")
        with self.assertRaises(RuntimeError):
            leitor.ler_particoes(caminho)

    def test_garantir_que_lanca_excessao_caso_nao_seja_diretorio(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "sem-particoes")
        caminho += "/arquivo.txt"
        with self.assertRaises(RuntimeError):
            leitor.ler_particoes(caminho)

if __name__ == "__main__":
    unittest.main()
