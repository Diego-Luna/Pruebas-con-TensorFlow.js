let net;

const imgEl = document.getElementById('img');
const desEl = document.getElementById('descripcion_img');

async function app() {
  net = await mobilenet.load();

  var result = await net.classify(imgEl);

  console.log(result);
  displayImagePredicction();
}

imgEl.onload = async function () {
  displayImagePredicction();
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