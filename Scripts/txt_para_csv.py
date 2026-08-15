#!/usr/bin/env python3
"""
Converte um arquivo .txt em um .csv de 3 colunas.

Uso básico (delimitador padrão = espaço/tab, detecta automaticamente):
    python txt_para_csv.py entrada.txt saida.csv

Especificando um delimitador (ex: vírgula, ponto e vírgula, pipe):
    python txt_para_csv.py entrada.txt saida.csv --delimitador ";"

Definindo nomes de colunas:
    python txt_para_csv.py entrada.txt saida.csv --colunas col1 col2 col3

Como funciona:
    Cada linha do .txt é dividida em até 3 partes usando o delimitador.
    - Se a linha tiver mais de 3 partes, as partes extras são juntadas na 3ª coluna.
    - Se tiver menos de 3 partes, as colunas faltantes ficam vazias.
    - Linhas totalmente vazias são ignoradas.
"""

import argparse
import csv
import re
import sys


def dividir_linha(linha: str, delimitador: str | None) -> list[str]:
    """Divide uma linha em até 3 partes."""
    linha = linha.strip("\n").strip("\r")

    if delimitador is None:
        # Sem delimitador definido: separa por qualquer sequência de espaços/tabs
        partes = re.split(r"\s+", linha.strip(), maxsplit=2)
    else:
        partes = linha.split(delimitador, maxsplit=2)
        partes = [p.strip() for p in partes]

    # Garante exatamente 3 colunas (preenche com "" se faltar)
    while len(partes) < 3:
        partes.append("")

    return partes[:3]


def converter(caminho_txt: str, caminho_csv: str, delimitador: str | None, colunas: list[str]):
    with open(caminho_txt, "r", encoding="utf-8") as f_in:
        linhas = [l for l in f_in.readlines() if l.strip() != ""]

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(colunas)
        for linha in linhas:
            writer.writerow(dividir_linha(linha, delimitador))

    print(f"Concluído! {len(linhas)} linha(s) convertida(s) para '{caminho_csv}'.")


def main():
    parser = argparse.ArgumentParser(description="Converte um .txt em um .csv de 3 colunas.")
    parser.add_argument("entrada", help="Caminho do arquivo .txt de entrada")
    parser.add_argument("saida", help="Caminho do arquivo .csv de saída")
    parser.add_argument(
        "--delimitador",
        default=None,
        help="Caractere usado para separar as colunas no .txt (padrão: qualquer espaço/tab)",
    )
    parser.add_argument(
        "--colunas",
        nargs=3,
        default=["coluna1", "coluna2", "coluna3"],
        help="Nomes das 3 colunas do CSV (padrão: coluna1 coluna2 coluna3)",
    )
    args = parser.parse_args()

    try:
        converter(args.entrada, args.saida, args.delimitador, args.colunas)
    except FileNotFoundError:
        print(f"Erro: arquivo '{args.entrada}' não encontrado.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
