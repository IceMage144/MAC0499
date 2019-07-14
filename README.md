## MAC0499 - Trabalho de conclusão de curso

### Rodando a primeira vez

Clone este repositório usando o comando

```
git clone https://github.com/IceMage144/MAC0499
```

Baixe a versão 3.0 do Godot, adicione este projeto à lista de projetos, abra o projeto (ignore qualquer aviso de erro do projeto), instale o plugin PythonScript através da aba AssetLib dentro do editor e reinicie o editor.

Depois de instalado, o plugin criará uma pasta pythonscript dentro da pasta do projeto, que contém pastas com interpretadores de Python para vários sistemas operacionais. Esses interpretadores têm o comando `pip` disponível porém não têm a biblioteca `libssl`, essencial para que o instalador do `pip` funcione. Copie os arquivos `libssl.a`, `libssl.so` e `libssl.so.1.0.0` da pasta `MAC0499/FirstInstall/libssl` para a pasta `MAC0499/pythonscript/<pasta do seu SO>/lib` (isso só foi testado no Linux, se não funcionar para seu SO, ache os arquivos `libssl` em alguma instalação externa do Python 3.6 em seu computador e copie-os para a pasta correspondente dentro da pasta `MAC0499/pythonscript/<pasta do seu SO>`).

Dentro da pasta correspondente ao seu sistema operacional, crie um viretualenv do python para gerenciar melhor as bibliotecas Python, usando o comando

```
virtualenv MAC0499/pythonscript/<pasta do seu SO>
```

Ative o virtualenv usando o comando

```
source MAC0499/pythonscript/<pasta do seu SO>/bin/activate
```

Verifique que o caminho do comando `pip` mudou para `<caminho para o projeto>/MAC0499/pythonscript/<pasta do seu SO>/bin/python` usando o comando `which pip`. Instale as bibliotecas Python necessárias para do projeto usando o comando

```
pip install -r MAC0499/FirstInstall/requirements.txt
```

Desative o virtualenv usando o comando

```
deactivate
```

### Acelerando o processamento do jogo

O Godot conta o tempo dentro do jogo através do número de quadros mostrados na tela. Por padrão a cada 60 quadros mostrados na tela se passa 1 segundo dentro do jogo. Se algo no jogo demorar mais que 1/60 segundo em tempo real para ser processado, os relógios do jogo ficam "atrasados". O Godot fornece uma flag para mudar o número de quadros que o jogo considera 1 segundo, consequentemente aumentando a velocidade do jogo (mas o tempo real entre quadros continua 1/60 segundo).

Suponha que o comando `godot` rode o executável do Godot. Se você quiser rodar o jogo 2 vezes mais rápido, basta diminuir o número de quadros que representam um segundo para 30 rodando o comando abaixo dentro da pasta do jogo

```
godot --fixed-fps 30
```

### Rodando o jogo headless

Suponha denovo que o comando `godot` rode o executável do Godot. Para rodar o jogo headless basta executar o seguinte comando dentro da pasta do jogo

```
godot --disable-render-loop
```

### Compilando o jogo

Atualmente não há um jeito fácil de compilar jogos que usam o plugin PythonScript.