var live = true;


function ApplyGradCam(tmpLastPrediction = 1){
    selectedModelMeta = GetCurrentModelMeta()
    cb = function(arg_data){
        done = arg_data[1];
        returnValues = arg_data[2];
        EnableDisableStuff(true)
        if (done){
            ShowGradCam(returnValues)
            EnableDisableStuff(false)
        }else{
            ShowMessage(returnValues)
        }
    }
    pixels = GetImagePixels(selectedModelMeta.w, selectedModelMeta.h, selectedModelMeta.d)
    if (lastPrediction == null){
        lastPrediction = tmpLastPrediction
    }
    Worker_PostMessage("GRADCAM", [lastPrediction, lastImage], cb)
}

function ShowGradCam(arg_input){
    lastGradCam = arg_input;
    gradCamdiv.show();
    imageTensor = tf.tensor(arg_input[0])

    tf.browser.toPixels(imageTensor, gradCamCanvasOverlay)
    tf.browser.toPixels(imageTensor, gradCamCanvas)
    // tf.browser.toPixels(imageTensor, gradCamCanvas).then(() => {
    //     tf.browser.toPixels(imageTensor, gradCamCanvasOverlay).then(() => {
    //         imageTensor.dispose();
    //         console.log(
    //           "Make sure we cleaned up",
    //           tf.memory().numTensors
    //         );
    //       });
    //   });
}

function GetImagePixels(w, h, d){
    pixelsTensor = tf.browser.fromPixels(input_img[0]).resizeBilinear([w, h])
    pixels = pixelsTensor.reshape([1,w, h, d]).div(tf.scalar(255)).arraySync()
    lastImage = pixels
    pixelsTensor.dispose()
    return pixels
}

function Predict(){
    
    predictiondiv.hide()
    selectedModelMeta = GetCurrentModelMeta()
    cb = function(arg_data){
        done = arg_data[1];
        returnValues = arg_data[2];
        EnableDisableStuff(true)
        if (done){
            ShowResult(returnValues[0], returnValues[1])
            EnableDisableStuff(false)
        }else{
            ShowMessage(returnValues)
        }
    }
    pixels = GetImagePixels(selectedModelMeta.w, selectedModelMeta.h, selectedModelMeta.d)
    Worker_PostMessage("PREDICT", pixels, cb)
}

function ShowResult(arg_chance, arg_indexofpredicted){
    gradCamdiv.hide()
    predictiondiv.show()
    predictiondiv.find("#prediction_text").html(labels[arg_indexofpredicted] + " - " + (arg_chance* 100).toFixed(2)  + "%")
    lastPrediction = arg_indexofpredicted;
}

function SelectModel(){
    predictiondiv.hide()
    cb = function(arg_data){
        done = arg_data[1];
        returnValues = arg_data[2];
        
        ShowMessage(returnValues)
        EnableDisableStuff(true)
        if (done){
            selectedModel = mdlSelect.val()
            EnableDisableStuff(false)
        }
    }
    Worker_PostMessage("SELECT_MODEL", mdlSelect.val(), cb)
}

function SetupWorker(){
    if (window.Worker) {
        predictWorker = new Worker("worker.js");
        predictWorker.responses = {}
        
       
        predictWorker.onmessage = function(e) {
            // console.log('Message received from worker');
            console.log("Main", e.data);
            topic = e.data[0];
            done = e.data[1];
            returnValues = e.data[2];
            submitData = predictWorker.responses[topic];
            if (submitData == null){
                return
            }
            submitData['answered'] = done;
            if (submitData['callback'] != null){
                predictWorker.responses[topic]['callback'](e.data)
            }
            

        }
        Worker_PostMessage("SET_MODEL_JS", models)
    }else{
        alert("Este navegador não é compatível com a página atual.")
    }
}

function GetCurrentModelMeta(){
    if (selectedModel == null || models[selectedModel] == null) return false;
    return models[selectedModel];
}

function LoadModelList(){
    mdlSelect.empty()
    $.each(models, function(i, obj) {   
        mdlSelect.append($("<option></option>").attr("value", i).text(GetObjName(obj))); 
    });
    mdlSelect.attr("disabled", false);


    selectImage.empty()
    selectImage.append($("<option></option>").attr("value", -1).attr("disabled", "disabled").text("--------- SELECIONE UMA IMAGEM ---------")); 
    $.each(test_images, function(i, obj) {   
        selectImage.append($("<option></option>").attr("value", i).text(test_images[i].target)); 
    });
    selectImage.attr("disabled", false);
    selectImage.val(-1)
}




var selectedModel = null;
var predictWorker = null
var lastPrediction = null
var lastImage = null
var lastGradCam = null


function readImageToDiv(file, arg_div) {
    // Check if the file is an image.
    if (file == undefined){
      arg_div.hide()
      arg_div.removeAttr("src")
      EnableDisableStuff(false)
      return
    }
    if (file.type && !file.type.startsWith('image/')) {
      console.log('File is not an image.', file.type, file);
      arg_div.hide()
      arg_div.removeAttr("src")
      EnableDisableStuff(false)
      return
    }
    EnableDisableStuff(true)
    window.fileToPredict = file;
    const reader = new FileReader();
    reader.addEventListener('load', (event) => {
      arg_div[0].src = event.target.result;
      arg_div.css('display','block');
      EnableDisableStuff(false)
    });
    reader.readAsDataURL(file);
}
  
function ShowMessage(msg, tempo=3000){
      Toastify({
        text: msg,
        duration: tempo,
        gravity: "top", // `top` or `bottom`
        close: true,
        position: "center", // `left`, `center` or `right`
        stopOnFocus: true, // Prevents dismissing of toast on hover
      }).showToast();
}
  
function GetObjName(obj){
      return "[{0} {1}] {2} ({3}) - {4}x{5}x{6}".formatUnicorn(obj.day, obj.time, obj.name, obj.variation, obj.w, obj.h, obj.d)
}

function changeInput(){
    EnableDisableStuff(true)
    curType = $('input[name="inlineRadioOptions"]:checked').val()
    $(".custom-file-input").val("")
    selectImage.val(-1)
    input_img.removeAttr("src")
    input_img.hide()
    console.log(curType)
    if (curType == "upload"){
        $("#row_upload").show()
        $("#row_selectimage").hide()
        
    }else{
        $("#row_upload").hide()
        $("#row_selectimage").show()
    }
    EnableDisableStuff(false)

}


function Worker_PostMessage(msg_command, data, arg_callback = null) {
    predictWorker.responses[msg_command] = {"answered":false, "callback": arg_callback};
    predictWorker.postMessage([msg_command, data]);
    
}

function EnableDisableStuff(loadingAnything){
    if (loadingAnything){
        
        $('.loading').show();
        ToggleDisabled($("#row_modelo"), false)
        ToggleDisabled($("#row_change_select"), false)
        ToggleDisabled($("#row_upload"), false)
        ToggleDisabled($("#row_selectimage"), false)
        ToggleDisabled($("#row_submit"), false)
        ToggleDisabled($("#prediction"), false)
        return
    }
    
    $('.loading').hide();
    ToggleDisabled($("#row_modelo"), true)
    ToggleDisabled($("#row_upload"), true)
    ToggleDisabled($("#row_change_select"), true)
    ToggleDisabled($("#row_selectimage"), true)
    if (input_img[0].src == "" || selectedModel == null){
        ToggleDisabled($("#row_submit"), false)
        ToggleDisabled($("#prediction"), false)
    }else{
        ToggleDisabled($("#row_submit"), true)
        ToggleDisabled($("#prediction"), true)
    }
}

function ToggleDisabled(main_obj, target){
    disableFunc = function () {
        $(this).attr('disabled', 'disabled');
    }
    
    enableFunc = function () {
        $(this).removeAttr('disabled');
    }
    $(main_obj).find('input, select, file, button').each(target ? enableFunc : disableFunc);
}


String.prototype.formatUnicorn = String.prototype.formatUnicorn || function () {
    "use strict";
    var str = this.toString();
    if (arguments.length) {
        var t = typeof arguments[0];
        var key;
        var args = ("string" === t || "number" === t) ?
            Array.prototype.slice.call(arguments)
            : arguments[0];

        for (key in args) {
            str = str.replace(new RegExp("\\{" + key + "\\}", "gi"), args[key]);
        }
    }

    return str;
};


