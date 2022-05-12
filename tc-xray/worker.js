importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.5.2/dist/tf.min.js')

onmessage = function(e) {
    console.log('Worker: Message received from main script');
    console.log(e.data)
    switch(e.data[0]){
        case "SET_MODEL_JS":
            currentModelsJS = e.data[1];
            ReturnMessage(e.data[0], "Definido os metadados dos modelos no worker!");
            break;
        case "SELECT_MODEL":
            SelectModel(e.data[0], e.data[1]);
            break;
        case "PREDICT":
            returnValues = PredictInFact(e.data[0], e.data[1]);
            ReturnMessage(e.data[0], "Retornando valor predito...", false);
            ReturnMessage(e.data[0], returnValues);
            break;
    }
  }

function ReturnMessage(baseMsg, value, shouldEnd = true){
    postMessage([baseMsg, shouldEnd, value]);
}

function PredictInFact(baseMsg, arg_inputlist){
    // ReturnMessage(baseMsg, "Pegando modelo atual...", false);
    
    esteModeloMeta = GetCurrentModelMeta();
    if (esteModeloMeta == null){ 
        ReturnMessage(baseMsg, "Não foi possível carregar os metadados desse modelo...");
        return
    }
    esteModelo = GetCurrentModel();
    if (esteModelo == null){ 
        ReturnMessage(baseMsg, "Primeiro selecione um modelo a ser baixado...");
        return
    }
    // ReturnMessage(baseMsg, "Encontrado modelo atual, formatando a imagem selecionada...", false);
    tensorList = [tf.tensor(arg_inputlist).reshape([1, esteModeloMeta.w, esteModeloMeta.h, esteModeloMeta.d])]
    ReturnMessage(baseMsg, "Encontrada e formatada a imagem, tentando predizer a imagem selecionada...", false);
    predicted = esteModelo.predict(tensorList);
    ReturnMessage(baseMsg, "Sincronizando respostas...", false);
    predictedTensor = predicted.dataSync()
    indexOfPredicted = predictedTensor.indexOf(Math.max(...predictedTensor));
    return [predictedTensor[indexOfPredicted], indexOfPredicted]
}

function SelectModel(baseMsg, model_id){
    if (currentModels[model_id] != null){
        selectedModel = model_id;
        ReturnMessage(baseMsg, "Modelo selecionado.");
        return;
    }
    esteModeloMeta = currentModelsJS[model_id]
    if (esteModeloMeta == null){ 
        ReturnMessage(baseMsg, "Não foi possível carregar os metadados desse modelo...");
        return
    }
    ReturnMessage(baseMsg, "Iniciando download do modelo... '{0}'".formatUnicorn("{0} ({1})".formatUnicorn(esteModeloMeta.name, esteModeloMeta.variation)), false);
    tf.loadGraphModel('models/' + esteModeloMeta.folder + '/model.json').then(function(model) {
        currentModels[model_id] = model;
        selectedModel = model_id;
        ReturnMessage(baseMsg, "Modelo baixado com sucesso!");
    });
};

function GetCurrentModelMeta(){
    if (selectedModel == null || currentModelsJS[selectedModel] == null) return false;
    return currentModelsJS[selectedModel];
}

function GetCurrentModel(){
    if (selectedModel == null || currentModels[selectedModel] == null) return false;
    return currentModels[selectedModel];
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


var currentModelsJS = null;
var currentModels = {};
var selectedModel = null;