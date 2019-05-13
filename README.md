# Car-Detection-With-Tensorflow
Otomobil tespit etmek için Tensorflow Object Detection Api'si ile geliştirilmiş Convolutional Neural Network(CNN) sınıflandırıcısı.
Modelin oluşturulması için aşağıdaki adımların izlenmesi gerekmektedir.

### İlgili paper:
https://docs.google.com/document/d/1kAdHCibDpcTsyElk78JKOTX1ukuEa1TNrlgK3ApKRo0/edit

### Gereksinimler:
- fotoğrafları xml olarak etiketlemek için => LabelImg
- tensorflow object detection api modelleri için => git clone https://github.com/tensorflow/models.git
- tensorflow kurulumu için tensorflow/models/research dizininde setup yapılması => python setup.py install
- proto dosyalarının python kodlarına çevrimi için => https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip
- tensorflow tutorial dosyalarının kullanımı için => pip install jupyter
- kamera akışında sınıflandırma yapmak için => pip install opencv-python
- pip install matplotlib
- pip install pillow
- pip install lxml

## Car-Detection-With-Tensorflow klonlanması/indirilmesi

```
git clone https://github.com/dogabaris/Car-Detection-With-Tensorflow.git

```

## ssd_mobilenet_v1_pets indirilmesi
Pretrained ssd_mobilenet_v1_pets model checkpointiyle ve configiyle yeni model geliştirmesi yapabilmek için projenin klonlandığı dizinde açılan konsola

```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
```
yazılarak pretrained modelin indirilmesi ve arşivinden çıkarılması gerekmektedir.

## Modelin eğitimine başlanması
Tensorflow Modellerinini olduğu yer/research/object_detection dizinine Car-Detection-With-Tensorflow içeriğinin kopyalanması gerekmektedir.
Daha sonra dizinden açılan konsola aşağıdaki komut girilerek eğitime başlanmalıdır.

```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```

## Eğitimin takip edilmesi
Eğitim sürerken değerlerin takibi için konsola aşağıdaki komut girilerek TensorBoard aktif edilmeli ve 127.0.0.1:6006 adresinden izlenmelidir.

```
tensorboard --logdir='training'
```
oluşturulan checkpointler /training klasöründe bulunmaktadır.

## Eğitilen modelin frozen_inference_graph'a dönüştürülmesi
Eğitim durdurulduktan sonra modelin görmediği otomobil fotoğraflarıyla denenmesi için dondurulmuş sonuç grafının çıkarılmasına ihtiyaç vardır.

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix training/model.ckpt-(checkpointlerdeki en büyük sayı yazılmalı) --output_directory car_inference_graph
```
### no module named 'nets' hatası alınıyorsa
Sistem path'ine (tensorflow models'in bulunduğu dizin) models-master\research\object_detection ve 
models-master\research\object_detection\models eklenmelidir. Linux kullanılıyorsa tensorflow/models/ klasöründe açılan konsolda
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
yazılması gerekmektedir.
## Sonuç grafının kullanılması için object detection api örnek kodunun değiştirilmesi
Tensorflow object detection kodu(object_detection_tutorial.ipynb) dizinde açılan konsola aşağıdaki kod yazılarak jupyter notebook ile açılmalıdır.
```
jupyter notebook
```
Web tarayıcısında çalışan jupyter not defteri ile gösterilen dizinde object_detection_tutorial.ipynb kodunun tıklanıp açılması gerekmektedir.
Açılan sayfadaki kodta bir model indirilip object_detection/test_images klasörüne eklenen fotoğrafları sınıflandırılıyor. Bu kodun Variables başlığı altındaki kodu aşağıdaki kodla değiştirilmelidir.
```
# What model to download.
MODEL_NAME = 'car_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1
```
Download Model başlığı altındaki içerik silinmelidir. 
Detection başlığı altında test_images klasörü açılıp kodtaki aralıktaki images(fotoğraf numarası).jpg fotoğrafları alınarak modelde test edilmektedir.
```
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 8) ]
```
range parantez içi modele gönderilecek görüntülerin aralığını ifade etmektedir. Test edilmek istenen fotoğraflar image3.jpg, image4.jpg vb. sıralı şekilde eklenerek ve aralık düzenlenerek model test edilebilir.
Üstteki Cell sekmesinde Run All yapıldığında model fotoğrafları alır ve eğitildiği cismi fotoğraf içerisinde bulur ve cismi yüzde olarak benzettiği sınıfla kare içine alır.
