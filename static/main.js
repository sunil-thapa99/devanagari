var colour = '#000000';
var $canvas = $('canvas');

var context = $canvas[0].getContext('2d')
context.fillStyle = "#ffffff";
context.fillRect(0, 0, $canvas[0].width, $canvas[0].height);

var mouseDown = false;
var lastEvent;

// On mouse events on the canvas
$canvas.mousedown(function (e) {
    lastEvent = e;
    mouseDown = true;
}).mousemove(function (e) {
    // Draw lines
    if (mouseDown) {
        context.beginPath();
        context.moveTo(lastEvent.offsetX, lastEvent.offsetY);
        context.lineTo(e.offsetX, e.offsetY);
        context.strokeStyle = colour;
        context.lineWidth = 10;
        context.lineCap = 'round';
        context.stroke();
        lastEvent = e;
    }
}).mouseup(function () {
    mouseDown = false;
}).mouseleave(function () {
    $canvas.mouseup();
});

function clear_canvas() {
    context.clearRect(0,0,context.canvas.width,context.canvas.height);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, $canvas[0].width, $canvas[0].height);
}

function dataURItoBlob (dataURI) {
    // convert base64/URLEncoded data component to raw binary data held in a string
    var byteString;
    if (dataURI.split(',')[0].indexOf('base64') >= 0)
        byteString = atob(dataURI.split(',')[1]);
    else
        byteString = unescape(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to a typed array
    var ia = new Uint8Array(byteString.length);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ia], {type: mimeString});
}

function recog(){
    var url = $canvas[0].toDataURL();
    var newImg = document.getElementById('image_display');
    newImg.src = url;
    $('#solve_section').css('display', 'block')
    // var files = $('#file')[0].files;

    var blob = dataURItoBlob(url);
    var formData = new FormData(document.forms[0])
    formData.append("file", blob);
    // var files = $('#image_field')[0].files;

    // for(let i=0; i<files.length; i++) {
    //     formData.append("file", files[i]);
    // }

    // formData.append("file", url);
    // console.log(formData)
    $.ajax({
        url: 'solve',
        type: 'POST',
        data: formData,
        cache: false,
        processData: false,
        contentType: false,
        error: function(data){
            console.log("upload error" , data);
            console.log(data.getAllResponseHeaders());
        },
        success: function(data){
            console.log(data);
            // if (data['status'] == 'failed'){
            //     alert('Please insert valid image')
            // }else{
            //     $('#solve_section').attr("style", "display:block")
            
            //     bytestring = data['solved']
            //     image = bytestring.split('\'')[1]
            //     imagebox.attr('src' , 'data:image/jpeg;base64,'+image)

            //     bytestring = data['processed']
            //     image = bytestring.split('\'')[1]
            //     $('#processed_image').attr('src' , 'data:image/jpeg;base64,'+image)
            // }
        }
    });
}