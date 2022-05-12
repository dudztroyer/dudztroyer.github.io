

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

function GetImageArray(arg_width, arg_height, do_callback){
    thisImage = new Image()
    thisImage.onload = function(){
        canvas.width=arg_width
        canvas.height=arg_height
        context.drawImage(thisImage, 0, 0, arg_width, arg_height);
        data = context.getImageData(0, 0, arg_width, arg_width).data;
        var input = [];
        for(var i = 0; i < data.length; i += 4) {
            pixels = [data[i] / 255, data[i+1] / 255, data[i+2] / 255]
            input.push(pixels);
        }
        do_callback(input);
    }
    thisImage.src = input_img.attr("src");
}

function _base64ToArrayBuffer(base64) {
    var binary_string = window.atob(base64);
    var len = binary_string.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) {
        bytes[i] = binary_string.charCodeAt(i);
    }
    return bytes.buffer;
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




function Predict(){
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


    var predictCb = function(arg_input_data){
        
        Worker_PostMessage("PREDICT", arg_input_data, cb)
    }
    GetImageArray(selectedModelMeta.w, selectedModelMeta.h, predictCb)
}

function SelectModel(){
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

function Worker_PostMessage(msg_command, data, arg_callback = null) {
    predictWorker.responses[msg_command] = {"answered":false, "callback": arg_callback};
    predictWorker.postMessage([msg_command, data]);
    
}

function SetupWorker(){
    if (window.Worker) {
        predictWorker = new Worker("worker.js");
        predictWorker.responses = {}
        
        // first.onchange = function() {
        //   predictWorker.postMessage([first.value, second.value]);
        //   console.log('Message posted to worker');
        // }
        
        predictWorker.onmessage = function(e) {
            console.log('Message received from worker');
            console.log(e.data);
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
    }
}



function ShowResult(arg_chance, arg_indexofpredicted){
    predictiondiv.show()
    predictiondiv.find(".container").html(labels[arg_indexofpredicted] + " - " + (arg_chance* 100).toFixed(2)  + "%")
}

String.prototype.formatUnicorn = String.prototype.formatUnicorn ||
function () {
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

function ToggleDisabled(main_obj, target){
    disableFunc = function () {
        $(this).attr('disabled', 'disabled');
    }
    
    enableFunc = function () {
        $(this).removeAttr('disabled');
    }
    $(main_obj).find('input, select, file, button').each(target ? enableFunc : disableFunc);
}

function GetCurrentModelMeta(){
    if (selectedModel == null || models[selectedModel] == null) return false;
    return models[selectedModel];
}

function EnableDisableStuff(loadingAnything){
    if (loadingAnything){
        ToggleDisabled($("#row_modelo"), false)
        ToggleDisabled($("#row_change_select"), false)
        ToggleDisabled($("#row_upload"), false)
        ToggleDisabled($("#row_selectimage"), false)
        ToggleDisabled($("#row_submit"), false)
        return
    }
    ToggleDisabled($("#row_modelo"), true)
    ToggleDisabled($("#row_upload"), true)
    ToggleDisabled($("#row_change_select"), true)
    ToggleDisabled($("#row_selectimage"), true)
    if (input_img[0].src == "" || selectedModel == null){
        ToggleDisabled($("#row_submit"), false)
    }else{
        ToggleDisabled($("#row_submit"), true)
    }
}

var selectedModel = null;
var predictWorker = null
