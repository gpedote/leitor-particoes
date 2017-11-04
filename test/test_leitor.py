#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import unittest

import numpy as np
import pandas as pd

from .context import Leitor

class TestParticaoForaDeOrdem(unittest.TestCase):

    def setUp(self):
        leitor = Leitor()
        caminho = os.path.join(os.path.dirname(__file__), "resources", "outras-particoes",
                "desordenada")
        self.particoes = leitor.ler_particoes(caminho)

    def test_garantir_ordem_labels(self):
        part = self.particoes.loc[0]
        labels1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

        labels2 = part["particao"]["label"]
        self.assertTrue(np.array_equal(labels1, labels2), "Labels fora de ordem")

    def test_garantir_ordem_ids(self):
        part = self.particoes.loc[0]
        ids1 = ["B-ALL-BCR-ABL1", "B-ALL-BCR-ABL10", "B-ALL-BCR-ABL11", "B-ALL-BCR-ABL12",
                "B-ALL-BCR-ABL13", "B-ALL-BCR-ABL14", "B-ALL-BCR-ABL15", "B-ALL-BCR-ABL2",
                "B-ALL-BCR-ABL3", "B-ALL-BCR-ABL4", "B-ALL-BCR-ABL5", "B-ALL-BCR-ABL6",
                "B-ALL-BCR-ABL7", "B-ALL-BCR-ABL8", "B-ALL-BCR-ABL9", "B-ALL-E2A-PBX116",
                "B-ALL-E2A-PBX117", "B-ALL-E2A-PBX118", "B-ALL-E2A-PBX119", "B-ALL-E2A-PBX120",
                "B-ALL-E2A-PBX121", "B-ALL-E2A-PBX122", "B-ALL-E2A-PBX123", "B-ALL-E2A-PBX124",
                "B-ALL-E2A-PBX125", "B-ALL-E2A-PBX126", "B-ALL-E2A-PBX127", "B-ALL-E2A-PBX128",
                "B-ALL-E2A-PBX129", "B-ALL-E2A-PBX130", "B-ALL-E2A-PBX131", "B-ALL-E2A-PBX132",
                "B-ALL-E2A-PBX133", "B-ALL-E2A-PBX134", "B-ALL-E2A-PBX135", "B-ALL-E2A-PBX136",
                "B-ALL-E2A-PBX137", "B-ALL-E2A-PBX138", "B-ALL-E2A-PBX139", "B-ALL-E2A-PBX140",
                "B-ALL-E2A-PBX141", "B-ALL-E2A-PBX142", "B-ALL-HYP100", "B-ALL-HYP101",
                "B-ALL-HYP102", "B-ALL-HYP103", "B-ALL-HYP104", "B-ALL-HYP105", "B-ALL-HYP106",
                "B-ALL-HYP43", "B-ALL-HYP44", "B-ALL-HYP45", "B-ALL-HYP46", "B-ALL-HYP47",
                "B-ALL-HYP48", "B-ALL-HYP49", "B-ALL-HYP50", "B-ALL-HYP51", "B-ALL-HYP52",
                "B-ALL-HYP53", "B-ALL-HYP54", "B-ALL-HYP55", "B-ALL-HYP56", "B-ALL-HYP57",
                "B-ALL-HYP58", "B-ALL-HYP59", "B-ALL-HYP60", "B-ALL-HYP61", "B-ALL-HYP62",
                "B-ALL-HYP63", "B-ALL-HYP64", "B-ALL-HYP65", "B-ALL-HYP66", "B-ALL-HYP67",
                "B-ALL-HYP68", "B-ALL-HYP69", "B-ALL-HYP70", "B-ALL-HYP71", "B-ALL-HYP72",
                "B-ALL-HYP73", "B-ALL-HYP74", "B-ALL-HYP75", "B-ALL-HYP76", "B-ALL-HYP77",
                "B-ALL-HYP78", "B-ALL-HYP79", "B-ALL-HYP80", "B-ALL-HYP81", "B-ALL-HYP82",
                "B-ALL-HYP83", "B-ALL-HYP84", "B-ALL-HYP85", "B-ALL-HYP86", "B-ALL-HYP87",
                "B-ALL-HYP88", "B-ALL-HYP89", "B-ALL-HYP90", "B-ALL-HYP91", "B-ALL-HYP92",
                "B-ALL-HYP93", "B-ALL-HYP94", "B-ALL-HYP95", "B-ALL-HYP96", "B-ALL-HYP97",
                "B-ALL-HYP98", "B-ALL-HYP99", "B-ALL-MLL107", "B-ALL-MLL108", "B-ALL-MLL109",
                "B-ALL-MLL110", "B-ALL-MLL111", "B-ALL-MLL112", "B-ALL-MLL113", "B-ALL-MLL114",
                "B-ALL-MLL115", "B-ALL-MLL116", "B-ALL-MLL117", "B-ALL-MLL118", "B-ALL-MLL119",
                "B-ALL-MLL120", "B-ALL-MLL121", "B-ALL-MLL122", "B-ALL-MLL123", "B-ALL-MLL124",
                "B-ALL-MLL125", "B-ALL-MLL126", "B-ALL-TEL-AML1249", "B-ALL-TEL-AML1250",
                "B-ALL-TEL-AML1251", "B-ALL-TEL-AML1252", "B-ALL-TEL-AML1253",
                "B-ALL-TEL-AML1254", "B-ALL-TEL-AML1255", "B-ALL-TEL-AML1256",
                "B-ALL-TEL-AML1257", "B-ALL-TEL-AML1258", "B-ALL-TEL-AML1259",
                "B-ALL-TEL-AML1260", "B-ALL-TEL-AML1261", "B-ALL-TEL-AML1262",
                "B-ALL-TEL-AML1263", "B-ALL-TEL-AML1264", "B-ALL-TEL-AML1265",
                "B-ALL-TEL-AML1266", "B-ALL-TEL-AML1267", "B-ALL-TEL-AML1268",
                "B-ALL-TEL-AML1269", "B-ALL-TEL-AML1270", "B-ALL-TEL-AML1271",
                "B-ALL-TEL-AML1272", "B-ALL-TEL-AML1273", "B-ALL-TEL-AML1274",
                "B-ALL-TEL-AML1275", "B-ALL-TEL-AML1276", "B-ALL-TEL-AML1277",
                "B-ALL-TEL-AML1278", "B-ALL-TEL-AML1279", "B-ALL-TEL-AML1280",
                "B-ALL-TEL-AML1281", "B-ALL-TEL-AML1282", "B-ALL-TEL-AML1283",
                "B-ALL-TEL-AML1284", "B-ALL-TEL-AML1285", "B-ALL-TEL-AML1286",
                "B-ALL-TEL-AML1287", "B-ALL-TEL-AML1288", "B-ALL-TEL-AML1289",
                "B-ALL-TEL-AML1290", "B-ALL-TEL-AML1291", "B-ALL-TEL-AML1292",
                "B-ALL-TEL-AML1293", "B-ALL-TEL-AML1294", "B-ALL-TEL-AML1295",
                "B-ALL-TEL-AML1296", "B-ALL-TEL-AML1297", "B-ALL-TEL-AML1298",
                "B-ALL-TEL-AML1299", "B-ALL-TEL-AML1300", "B-ALL-TEL-AML1301",
                "B-ALL-TEL-AML1302", "B-ALL-TEL-AML1303", "B-ALL-TEL-AML1304",
                "B-ALL-TEL-AML1305", "B-ALL-TEL-AML1306", "B-ALL-TEL-AML1307",
                "B-ALL-TEL-AML1308", "B-ALL-TEL-AML1309", "B-ALL-TEL-AML1310",
                "B-ALL-TEL-AML1311", "B-ALL-TEL-AML1312", "B-ALL-TEL-AML1313",
                "B-ALL-TEL-AML1314", "B-ALL-TEL-AML1315", "B-ALL-TEL-AML1316",
                "B-ALL-TEL-AML1317", "B-ALL-TEL-AML1318", "B-ALL-TEL-AML1319",
                "B-ALL-TEL-AML1320", "B-ALL-TEL-AML1321", "B-ALL-TEL-AML1322",
                "B-ALL-TEL-AML1323", "B-ALL-TEL-AML1324", "B-ALL-TEL-AML1325",
                "B-ALL-TEL-AML1326", "B-ALL-TEL-AML1327", "OTHERS127", "OTHERS128",
                "OTHERS129", "OTHERS130", "OTHERS131", "OTHERS132", "OTHERS133", "OTHERS134",
                "OTHERS135", "OTHERS136", "OTHERS137", "OTHERS138", "OTHERS139", "OTHERS140",
                "OTHERS141", "OTHERS142", "OTHERS143", "OTHERS144", "OTHERS145", "OTHERS146",
                "OTHERS147", "OTHERS148", "OTHERS149", "OTHERS150", "OTHERS151", "OTHERS152",
                "OTHERS153", "OTHERS154", "OTHERS155", "OTHERS156", "OTHERS157", "OTHERS158",
                "OTHERS159", "OTHERS160", "OTHERS161", "OTHERS162", "OTHERS163", "OTHERS164",
                "OTHERS165", "OTHERS166", "OTHERS167", "OTHERS168", "OTHERS169", "OTHERS170",
                "OTHERS171", "OTHERS172", "OTHERS173", "OTHERS174", "OTHERS175", "OTHERS176",
                "OTHERS177", "OTHERS178", "OTHERS179", "OTHERS180", "OTHERS181", "OTHERS182",
                "OTHERS183", "OTHERS184", "OTHERS185", "OTHERS186", "OTHERS187", "OTHERS188",
                "OTHERS189", "OTHERS190", "OTHERS191", "OTHERS192", "OTHERS193", "OTHERS194",
                "OTHERS195", "OTHERS196", "OTHERS197", "OTHERS198", "OTHERS199", "OTHERS200",
                "OTHERS201", "OTHERS202", "OTHERS203", "OTHERS204", "OTHERS205", "T-ALL206",
                "T-ALL207", "T-ALL208", "T-ALL209", "T-ALL210", "T-ALL211", "T-ALL212",
                "T-ALL213", "T-ALL214", "T-ALL215", "T-ALL216", "T-ALL217", "T-ALL218",
                "T-ALL219", "T-ALL220", "T-ALL221", "T-ALL222", "T-ALL223", "T-ALL224",
                "T-ALL225", "T-ALL226", "T-ALL227", "T-ALL228", "T-ALL229", "T-ALL230",
                "T-ALL231", "T-ALL232", "T-ALL233", "T-ALL234", "T-ALL235", "T-ALL236",
                "T-ALL237", "T-ALL238", "T-ALL239", "T-ALL240", "T-ALL241", "T-ALL242",
                "T-ALL243", "T-ALL244", "T-ALL245", "T-ALL246", "T-ALL247", "T-ALL248"]

        ids2 = part["particao"]["id"]
        self.assertTrue(np.array_equal(ids1, ids2), "IDs fora de ordem")


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
