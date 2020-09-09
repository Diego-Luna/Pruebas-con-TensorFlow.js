let net;

const imgEl = document.getElementById('img');
const desEl = document.getElementById('descripcion_img');
const webcamElement = document.getElementById('webcamp');
const classiFier = knnClassifier.create();

async function app() {
  net = await mobilenet.load();

  var result = await net.classify(imgEl);

  console.log(result);
  displayImagePredicction();

  // para usar y ver la camara WebCamp
  webcam = await tf.data.webcam(webcamElement);

  while (true) {
    const img = await webcam.capture();

    const result = await net.classify(img);

    const activation = net.infer(img, "conv_preds");
    var result2;

    try {
      result2 = await classiFier.predicClass(activation);
      const classes = ['Untrained', 'Gato', 'Dino', 'Alex', 'Ok', 'Rock']
      // document.getElementById('console2' + classes[result2.label]);
      document.getElementById('console2').innerHTML = "Console2 prediction: " + classes[result2.label];

    } catch (error) {
      console.log(error);
    }

    document.getElementById('console').innerHTML = 'Prediction: ' + result[0].className + " <Probability" + result[0].probability

    // limpiamos los tensores de la memoria, sin saturar el navegador
    img.dispose();

    // esperamos al siguiente cuadro
    await tf.nextFrame();

  }

}

imgEl.onload = async function () {
  displayImagePredicction();
}

async function addExample(classId) {

  console.log('add example');

  const img = await webcam.capture();
  const activation = net.infer(img, true);

  // asemos el entrenamiento
  classiFier.addExample(activation, classId);

  // limpiamos la memoria
  img.dispose();
}

async function displayImagePredicction() {
  try {
    result = await net.classify(imgEl);
    desEl.innerHTML = JSON.stringify(result);
  } catch (error) {
    console.log(error);
  }
}

var count = 0;
async function cambiarImg() {
  count = count + 1;
  imgEl.src = 'https://picsum.photos/200/300?random=' + count;
}

app();