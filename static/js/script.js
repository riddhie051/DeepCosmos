
const imageInput =
document.getElementById("imageInput");

const previewImage =
document.getElementById("previewImage");

imageInput.addEventListener(
"change",

function(){


    const file =
    imageInput.files[0];

    if(file){

        previewImage.src =
        URL.createObjectURL(file);

        previewImage.style.display =
        "block";
    }
});
