# darknet-yolo repo
Docker image and scripts to fine-tuning a Yolo model in a custom dataset. (pt_BR)

---

## **OBS.:** Modelo `full` ainda não disponível no repositório. Utilizar apenas o `tiny`.

---

# Descrição

As instruções abaixo devem ser utilizadas para construir uma nova imagem Darknet para treinamento e refino de modelos Yolo.

Essa imagem deve ser utilizada somente em computadores que possuam placa GPU com drive cuda 10.x.

## Construindo a imagem docker com Darknet
```
docker build -t darknet-yolo .
```

## Testando a imagem docker criada
```
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -it darknet-yolo darknet --help
```

# Training

Para fazer fine-tuning em um dataset customizado, siga as instruções abaixo:

*Tips:*
- If you get CUDA out of memory adjust the subdivisions parameter.
- Adjust max_batches parameter down for shorter training time.
- How to improve the detection https://github.com/AlexeyAB/darknet#how-to-improve-object-detection

## Dataset
---

- Divida o dataset em `train`, `val` e `test`.
- Armazene as imagens dentro de `dataset/<ds_name>`
- Crie um arquivo `txt` com os labels.

A estrutura deve ficar como o exemplo abaixo:

```
dataset/cidade
├── classes.txt
├── test
│   ├── imagem_00038.jpg
│   ├── imagem_00038.txt
│   ├── imagem_00044.jpg
│   ├── imagem_00044.txt
│   ├── imagem_00386.jpg
│   └── imagem_00386.txt
├── train
│   ├── imagem_00001.jpg
│   ├── imagem_00001.txt
│   ├── imagem_00002.jpg
│   ├── imagem_00002.txt
│   ├── imagem_00003.jpg
│   ├── imagem_00003.txt
│   ├── imagem_00409.jpg
│   └── imagem_00409.txt
└── val
    ├── imagem_00000.jpg
    ├── imagem_00000.txt
    ├── imagem_00004.jpg
    └── imagem_00004.txt
```
Para cada imagem, tenha um arquivo `.txt` com o respectivo nome do arquivo da imagem.

Cada `.txt` deverá ter as informações dos bounding boxes existentes nas respectivas imagens. 

Para cada linha do arquivo, deverá seguir o seguinte formato:

```
<object-class> <x_center> <y_center> <width> <height>
```
Onde:
* `<object-class>` - integer object number from 0 to (classes-1)
* `<x_center> <y_center> <width> <height>` - float values relative to width and height of image, it can be equal from `(0.0 to 1.0]`
* for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
* atention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner)

Por exemplo, para a imagem `img1.jpg` você criará um arquivo `img1.txt` contendo:

```
1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
```

## Gerar os arquivos de configuração
---

Executar o script abaixo para gerar os arquivos de configuração necessários para o treinamento.

```
$ cd yolo/scripts
```
Para Yolo `full`:
```
$ python prepare_dataset.py -d ../dataset/<ds_name> -l classes.txt -y full
```

Para Yolo `tiny`:
```
$ python prepare_dataset.py -d ../dataset/<ds_name> -l classes.txt -y tiny
```

## Treinar o modelo
---

```
$ cd yolov4
$ docker run --rm -v `pwd`/yolo:/yolo -p 8090:8090 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -it darknet-yolo bash
$ cd /yolo
```

Para Yolo `full`:
```
darknet detector train /yolo/data/obj.data /yolo/cfg/yolo-obj-full.cfg /yolo/models/yolov4-full.conv.137 -dont_show -mjpeg_port 8090 -map
```

Para Yolo `tiny`:
```
darknet detector train /yolo/data/obj.data /yolo/cfg/yolo-obj-tiny.cfg /yolo/models/yolov4-tiny.conv.29 -dont_show -mjpeg_port 8090 -map
```

Para ver o gráfico de treinamento, abra o browser no endereço `http://127.0.0.1:8090`

# Test

Identificar o melhor modelo rodando os snapshots no dataset de `test` e verificando o que tem melhor resultado.

## Selecionando o melhor modelo
---

Após o término do treinamento, os arquivos `.weights` estarão armazedos dentro da pasta `yolo/backup` e devemos escolher o melhor deles.

Por exemplo, nosso treinamento pode ter parado em 9000 interações, mas o melhor resultado pode ter acontecido em arquivos de `.weights` anteriores (6000, 7000, 8000). Isso pode acontecer devido ao overfitting do modelo.

Overfitting é quando o modelo pode detectar objetos em imagens de treinamento, mais não consegue detectar em outras imagens, ou seja, o modelo não consegue generalizar. Para termos um detector de objetos mais genérico, devemos escolher uma versão dos arquivos `.weights` gerados que consiga generalizar mais, que basicamente são snapshots dos pesos gerados a cada 1000 interações.

Para escolher o melhor modelo, faça:

* Execute o comando abaixo para cada arquivo `backup\yolo-obj-[full|tiny]_?000.weights`

```
docker run --rm -v `pwd`/yolo:/yolo --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -it darknet-yolo darknet detector map /yolo/data/obj.data.test /yolo/cfg/yolo-obj-tiny.cfg /yolo/backup/yolo-obj-tiny_6000.weights
```

* Compare as últimas linhas de cada arquivo `.weights` (6000, 7000, 8000) e escolha arquivo `.weight` com o maior **mAP (mean average precision)** ou **IoU (intersect over union)**.

Por exemplo:
* **yolo-obj-tiny_6000.weights**: mean average precision (mAP@0.50) = 0.882851, or 88.29% 
* **yolo-obj-tiny_7000.weights**:  mean average precision (mAP@0.50) = 0.899423, or 89.94 %
* **yolo-obj-tiny_8000.weights**:  mean average precision (mAP@0.50) = 0.892679, or 89.27 %

* Realize um teste no dataset de `test` entre o `.weights` de maior `mAP` selecionado acima e o `bakcup\yolo-obj-tiny_best.weights`. Veja o que tem o melhor resultado e o escolha como o modelo final do seu detector de objetos.

## Verificando a performance no dataset de test
---

Execute o comando abaixo com o arquivo de `.weights` selecionado:
```
docker run --rm -v `pwd`/yolo:/yolo --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -it darknet-yolo darknet detector map /yolo/data/obj.data.test /yolo/cfg/yolo-obj-tiny.cfg /yolo/backup/yolo-obj-tiny_best.weights
```

## Detectando objetos em imagens
---

* Crie um ambiente virtual para python `mkvirtualenv yolov4 -p python3.7`

* Instale os pacotes necessários `pip install -r requirements.txt`

* Utilize o script `run_detector.py` em uma imagem ou em conjunto de imagens.

```
python run_detector.py -c ../cfg/yolo-obj-tiny.cfg -w ../backup/yolo-obj-tiny_best.weights -p ../dataset/cidade/test/imagem_00038.jpg
```
ou
```
python run_detector.py -c ../cfg/yolo-obj-tiny.cfg -w ../backup/yolo-obj-tiny_best.weights -p ../dataset/cidade/test -m
```

# Anotando imagens

Várias ferramentas podem ser utilizadas para anotar as imagens. Mas aqui estaremos utilizando o labelImg (https://github.com/tzutalin/labelImg) que já salva os arquivos `.txt` no padrão Yolo.

Se você ainda não criou o ambiente python virtual e instalou os pacotes do `requirements.txt`, siga os passos da sessão anterior para preparar o ambiente python e instalar o `labelImg`.

Instale algumas dependências:
```
sudo apt-get install pyqt5-dev-tools
```

Inicie o label image:
```
labelImg
labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

Faça a anotação das imagens.

# Referências
- https://github.com/AlexeyAB/darknet/wiki
- https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/
- https://colab.research.google.com/drive/1PWOwg038EOGNddf6SXDG5AsC8PIcAe-G#scrollTo=GNVU7eu9CQj3
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html