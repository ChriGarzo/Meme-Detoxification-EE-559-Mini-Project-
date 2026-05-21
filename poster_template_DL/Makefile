.PHONY: main clean FORCE

main: poster.pdf

poster.pdf: FORCE
	latexmk -xelatex -interaction=nonstopmode poster.tex

clean:
	latexmk -xelatex -C
