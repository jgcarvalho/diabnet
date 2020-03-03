## Diabnet

Este projeto surgiu a partir colaboração com o pesquisador Márcio?, do laboratório do Dr ???. Através deles obtivemos um dataset com 2344 pacientes dos quais 541 apresentam diabetes tipo 2. Além do diagnóstico (_label_) e de características como sexo, idade e diagnóstico dos pais (nem sempre disponível), recebemos 3 conjuntos de 1000 [SNPs](https://pt.wikipedia.org/wiki/Polimorfismo_de_nucleot%C3%ADdeo_%C3%BAnico) cada. Esses 3 conjuntos foram selecionados pelo Márcio de acordo com a correlação dos SNPs em relação ao diagnóstico e são:

1. conjunto de SNPs correlacionados
2. conjunto de SNPs não correlacionados
3. conjunto de SNPs selecionados aleatóriamente

Uma propriedade intessante desses dados é que uma parte dos pacientes possui vários diagnósticos em intervalos de 5 anos. Por exemplo, diagnósticos aos 30, 35 e 40 anos. Isso permite determinar aproximadamente qual a idade a doença se desenvolveu.

### Objetivo

Na posse desses dados, nosso objetivo foi construir um modelo capaz de predizer qual o risco de um paciente desenvolver diabetes tipo 2 ao longo da vida. 

Na prática, o modelo tem como entrada um conjunto determinado de SNPs do paciente, sua idade, sexo e o diagnóstico dos pais (este último, quando estiver disponível). O resultado do modelo são distribuições de probabilidades do indivíduo desenvolver diabetes tipo 2 ao longo de várias faixas etárias. Esses resultados podem então ser comparados com outros pacientes, por exemplo, com pessoas com mais de 60 anos que não desenvolveram diabetes (**negativos**) ou com jovens diabéticos para assim estimar qual o risco relativo do paciente.

### Resultados

#### Distribuição de probabilidades

#### Treinamento das redes neurais artificiais

#### Teste do modelo

#### Teste do modelo excluindo pacientes jovens negativos

Outros resultados:

- Avaliação da predição para famílias
- Avaliação da predição por núcleos familiares (grupos de pais + filhos)
- Importância dos atributos na predição.

### Métodos

O modelo escolhido foi uma rede neural artificial _feed-forward_. A primeira camada é composta por neurônios localmente conectados (_locally connected layer_) que transforma o input para $x_i$   

 Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/jgcarvalho/diabnet/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/jgcarvalho/diabnet/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
