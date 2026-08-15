#!/usr/bin/env python3
"""
Converte um .csv (de 3 colunas) de volta em um arquivo .txt.

Uso básico (junta as colunas com espaço, sem o cabeçalho):
    python csv_para_txt.py entrada.csv saida.txt

Especificando um delimitador entre as colunas (ex: tab, vírgula, pipe):
    python csv_para_txt.py entrada.csv saida.txt --delimitador "\t"

Incluindo a linha de cabeçalho no .txt:
    python csv_para_txt.py entrada.csv saida.txt --incluir-cabecalho

Ignorando colunas vazias na hora de montar a linha:
    (ativado por padrão; use --manter-vazias para desativar)
"""

import argparse
import csv
import sys


def montar_linha(campos: list[str], delimitador: str, manter_vazias: bool) -> str:
    if not manter_vazias:
        campos = [c for c in campos if c.strip() != ""]
    return delimitador.join(campos)


def converter(caminho_csv: str, caminho_txt: str, delimitador: str, incluir_cabecalho: bool, manter_vazias: bool):
    with open(caminho_csv, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.reader(f_in)
        linhas = list(reader)

    if not linhas:
        print("CSV vazio.", file=sys.stderr)
        sys.exit(1)

    cabecalho, dados = linhas[0], linhas[1:]

    with open(caminho_txt, "w", encoding="utf-8") as f_out:
        if incluir_cabecalho:
            f_out.write(montar_linha(cabecalho, delimitador, manter_vazias) + "\n")
        for linha in dados:
            f_out.write(montar_linha(linha, delimitador, manter_vazias) + "\n")

    print(f"Concluído! {len(dados)} linha(s) convertida(s) para '{caminho_txt}'.")


def main():
    parser = argparse.ArgumentParser(description="Converte um .csv de volta em .txt.")
    parser.add_argument("entrada", help="Caminho do arquivo .csv de entrada")
    parser.add_argument("saida", help="Caminho do arquivo .txt de saída")
    parser.add_argument(
        "--delimitador",
        default=" ",
        help="Texto usado para juntar as colunas em cada linha (padrão: espaço)",
    )
    parser.add_argument(
        "--incluir-cabecalho",
        action="store_true",
        help="Inclui a linha de cabeçalho (nomes das colunas) no .txt",
    )
    parser.add_argument(
        "--manter-vazias",
        action="store_true",
        help="Mantém colunas vazias na linha (por padrão elas são omitidas)",
    )
    args = parser.parse_args()

    # Permite passar "\t" na linha de comando como tab de verdade
    delimitador = args.delimitador.encode().decode("unicode_escape")

    try:
        converter(args.entrada, args.saida, delimitador, args.incluir_cabecalho, args.manter_vazias)
    except FileNotFoundError:
        print(f"Erro: arquivo '{args.entrada}' não encontrado.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
