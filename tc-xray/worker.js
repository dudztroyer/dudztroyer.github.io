importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.5.2/dist/tf.min.js')

onmessage = function(e) {
    // console.log('Worker: Message received from main script');
    console.log("Worker", e.data)
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
        case "GRADCAM":
            returnValues = DoGradCam(e.data[0], e.data[1]);
            ReturnMessage(e.data[0], "Retornando GradCAM...", false);
            ReturnMessage(e.data[0], returnValues);
            break;
        case "SUMMARY":
            esteModelo = GetCurrentModel();
            ReturnMessage(e.data[0], esteModelo.summary());
            break;
    }
  }

function ReturnMessage(baseMsg, value, shouldEnd = true){
    postMessage([baseMsg, shouldEnd, value]);
}


function DoGradCam(baseMsg, arg_inputDATA){
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
    
    lastPrediction = arg_inputDATA[0]
    lastImage = arg_inputDATA[1]
    // gCamModel = GetGradCamModel(baseMsg, esteModelo, lastPrediction)
    ReturnMessage(baseMsg, "Criando imagem GradCAM para o índice " + lastPrediction, false);

    // ReturnMessage(baseMsg, "Encontrado modelo atual, formatando a imagem selecionada...", false);
    imageTensor = tf.tensor(lastImage)
    // gradCamCreated = gradClassActivationMap(esteModelo, lastPrediction, imageTensor)
    gradCamCreated = CustomGradCAM(esteModelo, lastPrediction, imageTensor)
    ReturnMessage(baseMsg, "Criado imagem GradCAM. Sincronizando respostas...", false);
    predictedTensor = gradCamCreated.arraySync()
    gradCamCreated.dispose();
    console.log(predictedTensor)
    return predictedTensor
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
    arg_inputlist = tf.tensor(arg_inputlist)
    ReturnMessage(baseMsg, "Encontrada e formatada a imagem, tentando predizer a imagem selecionada...", false);
    predicted = esteModelo.predict(arg_inputlist);
    ReturnMessage(baseMsg, "Sincronizando respostas...", false);
    predictedTensor = predicted.dataSync()
    predicted.dispose();
    console.log(predictedTensor)
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

    tf.loadLayersModel('models_new/' + esteModeloMeta.folder + '/model.json').then(function(model) {
        ReturnMessage(baseMsg, "Modelo baixado com sucesso... Inicializando modelo...", false);
        const warmupResult = model.predict(tf.zeros([1,esteModeloMeta.w,esteModeloMeta.h,esteModeloMeta.d]))
        warmupResult.dataSync();
        warmupResult.dispose();

        currentModels[model_id] = model;
        selectedModel = model_id;
        ReturnMessage(baseMsg, "Modelo novo inicializado com sucesso!");

    });
};

function GetCurrentModelMeta(){
    if (selectedModel == null || currentModelsJS[selectedModel] == null) return null;
    return currentModelsJS[selectedModel];
}

function GetCurrentModel(){
    if (selectedModel == null || currentModels[selectedModel] == null) return null;
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


function CustomGradCAM(model, classIndex, input){
    // input = tf.tensor(lastImage)
    // classIndex = lastPrediction
    // folder = models[0].folder
    // model = await tf.loadLayersModel('models_new/' + folder + '/model.json')

    return tf.tidy(() => {
        let layerIndex = model.layers.length - 1;
        while (layerIndex >= 0) {
        if (model.layers[layerIndex].outputShape.length == 4) {
            break;
        }
        layerIndex--;
        }
        const lastConvLayer = model.layers[layerIndex];
        console.log(
            `Located last 4d layer of the model at ` +
            `index ${layerIndex}: layer type = ${lastConvLayer.getClassName()}; ` +
            `layer name = ${lastConvLayer.name}`);


        /////////// COMPUTE_HEATMAP

        // subModel1 = tf.model(
        //     {
        //         inputs:model.inputs,
        //         outputs: lastConvLayerOutput
        //     }
        // )

        // const newInput = tf.input({shape: lastConvLayerOutput.shape.slice(1)});
        // layerIndex++;
        // let y = newInput;
        // while (layerIndex < model.layers.length) {
        //     y = model.layers[layerIndex++].apply(y);
        // }

        // const subModel2 = tf.model({inputs: newInput, outputs: y});

        // convOutputs = subModel1.apply(input)

        // const convOuputsFunc = (arg_input) => subModel2.apply(arg_input, {training: true}).gather([classIndex], 1);
        
        // const gradFunction = tf.grad(convOuputsFunc);
        // grads = gradFunction(convOutputs);
        // grads = grads.div(grads.max().sub(grads.min()))
        // console.log(grads.shape)



        const lastConvLayerOutput = lastConvLayer.output;
        subModel1 = tf.model(
                {
                    inputs:model.inputs,
                    outputs: [lastConvLayerOutput, model.output]
                }
        )
        
        preds = subModel1.apply(input)
        convOutputs = preds[0]

        // predictions = preds[1].arraySync()[0]
        // loss = predictions[classIndex]

        // grads = tf.randomUniform([1, 7, 7, 512])
        grads = tf.ones([1, 7, 7, 512])

        castConvOutputs = tf.cast(convOutputs, "bool")
        castGrads = tf.cast(grads, "bool")
        guidedGrads = castConvOutputs.mul(castGrads).mul(grads)
        
        convOutputs = convOutputs.reshape(convOutputs.shape.slice(1))
        guidedGrads = guidedGrads.reshape(guidedGrads.shape.slice(1))


        weights = tf.mean(guidedGrads, axis=(0, 1))
        multed = tf.mul(weights, convOutputs)
        cam = tf.sum(multed, axis=-1)
        
        w = input.shape[2]
        h = input.shape[1]
        heatmap = tf.image.resizeBilinear(cam.reshape([cam.shape[0], cam.shape[1],1]), [w,h])
        
        numer = heatmap.sub(heatmap.min())
        
        denom = heatmap.max().sub(heatmap.min()).add(1e-8)
        heatmap = numer.div(denom)
        
        heatmapCM = applyColorMap(heatmap.reshape([1, heatmap.shape[0], heatmap.shape[1], heatmap.shape[2]]));
        console.log("Make sure we cleaned up",tf.memory().numTensors);
        return heatmapCM
    });

}

function gradClassActivationMap(model, classIndex, x, overlayFactor = 2.0) {
    // Try to locate the last conv layer of the model.
    let layerIndex = model.layers.length - 1;
    while (layerIndex >= 0) {
        if (model.layers[layerIndex].outputShape.length == 4) {
            break;
        }
        layerIndex--;
    }
    tf.util.assert(
        layerIndex >= 0, `Failed to find a convolutional layer in model`);
  
    const lastConvLayer = model.layers[layerIndex];
    console.log(
        `Located last convolutional layer of the model at ` +
        `index ${layerIndex}: layer type = ${lastConvLayer.getClassName()}; ` +
        `layer name = ${lastConvLayer.name}`);
  
    // Get "sub-model 1", which goes from the original input to the output
    // of the last convolutional layer.
    const lastConvLayerOutput = lastConvLayer.output;
    const subModel1 =
        tf.model({inputs: model.inputs, outputs: lastConvLayerOutput});
  
    // Get "sub-model 2", which goes from the output of the last convolutional
    // layer to the original output.
    const newInput = tf.input({shape: lastConvLayerOutput.shape.slice(1)});
    layerIndex++;
    let y = newInput;
    while (layerIndex < model.layers.length) {
      y = model.layers[layerIndex++].apply(y);
    }
    const subModel2 = tf.model({inputs: newInput, outputs: y});
  
    return tf.tidy(() => {
        // This function runs sub-model 2 and extracts the slice of the probability
        // output that corresponds to the desired class.
        const convOutput2ClassOutput = (input) =>
            subModel2.apply(input, {training: true}).gather([classIndex], 1);
        // This is the gradient function of the output corresponding to the desired
        // class with respect to its input (i.e., the output of the last
        // convolutional layer of the original model).
        const gradFunction = tf.grad(convOutput2ClassOutput);

        // Calculate the values of the last conv layer's output.
        const lastConvLayerOutputValues = subModel1.apply(x);
        // Calculate the values of gradients of the class output w.r.t. the output
        // of the last convolutional layer.
        gradValues = gradFunction(lastConvLayerOutputValues);
        gradValues = gradValues.div(gradValues.max().sub(gradValues.min()))
        // Pool the gradient values within each filter of the last convolutional
        // layer, resulting in a tensor of shape [numFilters].
        const pooledGradValues = tf.mean(gradValues, [0, 1, 2]);
        
        // castGrads = tf.cast(pooledGradValues, "bool")
        // castConvOutputs = tf.cast(lastConvLayerOutputValues, "bool")
        // guidedGrads = castGrads.mul(castConvOutputs).mul(pooledGradValues);

        // convOutputs = lastConvLayerOutputValues.reshape([7,7,512])
        // guidedGrads = guidedGrads.reshape([7,7,512])

        // weights = tf.mean(guidedGrads, axis=[0, 1]);
        // heatMap = tf.sum(tf.mul(weights, convOutputs), axis=-1);
        // heatMap = heatMap.expandDims(-1).expandDims(0);

        // Scale the convlutional layer's output by the pooled gradients, using
        // broadcasting.
        const scaledConvOutputValues = lastConvLayerOutputValues.mul(pooledGradValues);
        
        // // Create heat map by averaging and collapsing over all filters.
        let heatMap = scaledConvOutputValues.mean(-1);
        // // Discard negative values from the heat map and normalize it to the [0, 1]
        // // interval.
        heatMap = heatMap.relu();
        // heatMap = heatMap.div(heatMap.max()).expandDims(-1);
       
        heatMap = heatMap.div(heatMap.max().sub(heatMap.min()).add(1e-8)).expandDims(-1);
        // Up-sample the heat map to the size of the input image.
        heatMap = tf.image.resizeBilinear(heatMap, [x.shape[1], x.shape[2]]);

        // Apply an RGB colormap on the heatMap. This step is necessary because
        // the heatMap is a 1-channel (grayscale) image. It needs to be converted
        // into a color (RGB) one through this function call.
        heatMap = applyColorMap(heatMap);
        return heatMap
        // To form the final output, overlay the color heat map on the input image.
        heatMap = heatMap.mul(overlayFactor).add(x.div(255));
        return heatMap.div(heatMap.max());
    });
}


const RGB_COLORMAP = [0.26666668, 0.00392157, 0.32941177, 0.26666668, 0.00784314,
       0.33333334, 0.26666668, 0.01176471, 0.34117648, 0.27058825,
       0.01960784, 0.34509805, 0.27058825, 0.02352941, 0.3529412 ,
       0.27058825, 0.03137255, 0.35686275, 0.27450982, 0.03529412,
       0.36078432, 0.27450982, 0.04313726, 0.36862746, 0.27450982,
       0.04705882, 0.37254903, 0.27450982, 0.05490196, 0.38039216,
       0.2784314 , 0.05882353, 0.38431373, 0.2784314 , 0.06666667,
       0.3882353 , 0.2784314 , 0.07058824, 0.39607844, 0.2784314 ,
       0.07843138, 0.4       , 0.2784314 , 0.08235294, 0.40392157,
       0.2784314 , 0.08627451, 0.4117647 , 0.2784314 , 0.09411765,
       0.41568628, 0.28235295, 0.09803922, 0.41960785, 0.28235295,
       0.10196079, 0.42352942, 0.28235295, 0.10980392, 0.43137255,
       0.28235295, 0.11372549, 0.43529412, 0.28235295, 0.11764706,
       0.4392157 , 0.28235295, 0.1254902 , 0.44313726, 0.28235295,
       0.12941177, 0.44705883, 0.28235295, 0.13333334, 0.4509804 ,
       0.28235295, 0.13725491, 0.45490196, 0.2784314 , 0.14509805,
       0.45882353, 0.2784314 , 0.14901961, 0.4627451 , 0.2784314 ,
       0.15294118, 0.46666667, 0.2784314 , 0.15686275, 0.47058824,
       0.2784314 , 0.16470589, 0.4745098 , 0.2784314 , 0.16862746,
       0.47843137, 0.2784314 , 0.17254902, 0.48235294, 0.27450982,
       0.1764706 , 0.4862745 , 0.27450982, 0.18431373, 0.4862745 ,
       0.27450982, 0.1882353 , 0.49019608, 0.27450982, 0.19215687,
       0.49411765, 0.27058825, 0.19607843, 0.49803922, 0.27058825,
       0.20392157, 0.49803922, 0.27058825, 0.20784314, 0.5019608 ,
       0.27058825, 0.21176471, 0.5058824 , 0.26666668, 0.21568628,
       0.5058824 , 0.26666668, 0.22352941, 0.50980395, 0.2627451 ,
       0.22745098, 0.5137255 , 0.2627451 , 0.23137255, 0.5137255 ,
       0.2627451 , 0.23529412, 0.5176471 , 0.25882354, 0.23921569,
       0.5176471 , 0.25882354, 0.24313726, 0.52156866, 0.25882354,
       0.2509804 , 0.52156866, 0.25490198, 0.25490198, 0.5254902 ,
       0.25490198, 0.25882354, 0.5254902 , 0.2509804 , 0.2627451 ,
       0.5294118 , 0.2509804 , 0.26666668, 0.5294118 , 0.24705882,
       0.27058825, 0.5294118 , 0.24705882, 0.2784314 , 0.53333336,
       0.24313726, 0.28235295, 0.53333336, 0.24313726, 0.28627452,
       0.5372549 , 0.23921569, 0.2901961 , 0.5372549 , 0.23921569,
       0.29411766, 0.5372549 , 0.23921569, 0.29803923, 0.5372549 ,
       0.23529412, 0.3019608 , 0.5411765 , 0.23529412, 0.30588236,
       0.5411765 , 0.23137255, 0.3137255 , 0.5411765 , 0.23137255,
       0.31764707, 0.5411765 , 0.22745098, 0.32156864, 0.54509807,
       0.22745098, 0.3254902 , 0.54509807, 0.22352941, 0.32941177,
       0.54509807, 0.22352941, 0.33333334, 0.54509807, 0.21960784,
       0.3372549 , 0.54509807, 0.21960784, 0.34117648, 0.54901963,
       0.21568628, 0.34509805, 0.54901963, 0.21568628, 0.34901962,
       0.54901963, 0.21176471, 0.3529412 , 0.54901963, 0.21176471,
       0.35686275, 0.54901963, 0.20784314, 0.36078432, 0.54901963,
       0.20784314, 0.3647059 , 0.54901963, 0.20392157, 0.36862746,
       0.5529412 , 0.20392157, 0.37254903, 0.5529412 , 0.2       ,
       0.3764706 , 0.5529412 , 0.2       , 0.38039216, 0.5529412 ,
       0.19607843, 0.38431373, 0.5529412 , 0.19607843, 0.3882353 ,
       0.5529412 , 0.19215687, 0.39215687, 0.5529412 , 0.19215687,
       0.39607844, 0.5529412 , 0.19215687, 0.4       , 0.5529412 ,
       0.1882353 , 0.40392157, 0.5529412 , 0.1882353 , 0.40784314,
       0.5529412 , 0.18431373, 0.4117647 , 0.5529412 , 0.18431373,
       0.41568628, 0.5529412 , 0.18039216, 0.41960785, 0.5568628 ,
       0.18039216, 0.42352942, 0.5568628 , 0.18039216, 0.42745098,
       0.5568628 , 0.1764706 , 0.43137255, 0.5568628 , 0.1764706 ,
       0.43529412, 0.5568628 , 0.17254902, 0.4392157 , 0.5568628 ,
       0.17254902, 0.44313726, 0.5568628 , 0.17254902, 0.44705883,
       0.5568628 , 0.16862746, 0.4509804 , 0.5568628 , 0.16862746,
       0.45490196, 0.5568628 , 0.16470589, 0.45882353, 0.5568628 ,
       0.16470589, 0.4627451 , 0.5568628 , 0.16470589, 0.46666667,
       0.5568628 , 0.16078432, 0.47058824, 0.5568628 , 0.16078432,
       0.4745098 , 0.5568628 , 0.15686275, 0.47843137, 0.5568628 ,
       0.15686275, 0.47843137, 0.5568628 , 0.15686275, 0.48235294,
       0.5568628 , 0.15294118, 0.4862745 , 0.5568628 , 0.15294118,
       0.49019608, 0.5568628 , 0.15294118, 0.49411765, 0.5568628 ,
       0.14901961, 0.49803922, 0.5568628 , 0.14901961, 0.5019608 ,
       0.5568628 , 0.14901961, 0.5058824 , 0.5568628 , 0.14509805,
       0.50980395, 0.5568628 , 0.14509805, 0.5137255 , 0.5529412 ,
       0.14117648, 0.5176471 , 0.5529412 , 0.14117648, 0.52156866,
       0.5529412 , 0.14117648, 0.5254902 , 0.5529412 , 0.13725491,
       0.5294118 , 0.5529412 , 0.13725491, 0.53333336, 0.5529412 ,
       0.13725491, 0.5372549 , 0.5529412 , 0.13333334, 0.5372549 ,
       0.5529412 , 0.13333334, 0.5411765 , 0.5529412 , 0.13333334,
       0.54509807, 0.5529412 , 0.12941177, 0.54901963, 0.5529412 ,
       0.12941177, 0.5529412 , 0.54901963, 0.12941177, 0.5568628 ,
       0.54901963, 0.1254902 , 0.56078434, 0.54901963, 0.1254902 ,
       0.5647059 , 0.54901963, 0.1254902 , 0.5686275 , 0.54901963,
       0.12156863, 0.57254905, 0.54901963, 0.12156863, 0.5764706 ,
       0.54509807, 0.12156863, 0.5803922 , 0.54509807, 0.12156863,
       0.58431375, 0.54509807, 0.12156863, 0.5882353 , 0.54509807,
       0.11764706, 0.5921569 , 0.5411765 , 0.11764706, 0.59607846,
       0.5411765 , 0.11764706, 0.6       , 0.5411765 , 0.11764706,
       0.6       , 0.5411765 , 0.11764706, 0.6039216 , 0.5372549 ,
       0.11764706, 0.60784316, 0.5372549 , 0.11764706, 0.6117647 ,
       0.5372549 , 0.11764706, 0.6156863 , 0.53333336, 0.11764706,
       0.61960787, 0.53333336, 0.11764706, 0.62352943, 0.53333336,
       0.11764706, 0.627451  , 0.5294118 , 0.12156863, 0.6313726 ,
       0.5294118 , 0.12156863, 0.63529414, 0.5254902 , 0.12156863,
       0.6392157 , 0.5254902 , 0.1254902 , 0.6431373 , 0.52156866,
       0.1254902 , 0.64705884, 0.52156866, 0.12941177, 0.6509804 ,
       0.52156866, 0.12941177, 0.654902  , 0.5176471 , 0.13333334,
       0.654902  , 0.5176471 , 0.13725491, 0.65882355, 0.5137255 ,
       0.13725491, 0.6627451 , 0.50980395, 0.14117648, 0.6666667 ,
       0.50980395, 0.14509805, 0.67058825, 0.5058824 , 0.14901961,
       0.6745098 , 0.5058824 , 0.15294118, 0.6784314 , 0.5019608 ,
       0.15686275, 0.68235296, 0.49803922, 0.16078432, 0.6862745 ,
       0.49803922, 0.16470589, 0.6901961 , 0.49411765, 0.16862746,
       0.69411767, 0.49019608, 0.17254902, 0.69411767, 0.49019608,
       0.18039216, 0.69803923, 0.4862745 , 0.18431373, 0.7019608 ,
       0.48235294, 0.1882353 , 0.7058824 , 0.47843137, 0.19607843,
       0.70980394, 0.47843137, 0.2       , 0.7137255 , 0.4745098 ,
       0.20784314, 0.7176471 , 0.47058824, 0.21176471, 0.72156864,
       0.46666667, 0.21960784, 0.7254902 , 0.4627451 , 0.22352941,
       0.7254902 , 0.4627451 , 0.23137255, 0.7294118 , 0.45882353,
       0.23921569, 0.73333335, 0.45490196, 0.24313726, 0.7372549 ,
       0.4509804 , 0.2509804 , 0.7411765 , 0.44705883, 0.25882354,
       0.74509805, 0.44313726, 0.26666668, 0.74509805, 0.4392157 ,
       0.27058825, 0.7490196 , 0.43529412, 0.2784314 , 0.7529412 ,
       0.43137255, 0.28627452, 0.75686276, 0.42745098, 0.29411766,
       0.7607843 , 0.42352942, 0.3019608 , 0.7607843 , 0.41960785,
       0.30980393, 0.7647059 , 0.4117647 , 0.31764707, 0.76862746,
       0.40784314, 0.3254902 , 0.77254903, 0.40392157, 0.33333334,
       0.7764706 , 0.4       , 0.34117648, 0.7764706 , 0.39607844,
       0.34901962, 0.78039217, 0.39215687, 0.35686275, 0.78431374,
       0.38431373, 0.36862746, 0.7882353 , 0.38039216, 0.3764706 ,
       0.7882353 , 0.3764706 , 0.38431373, 0.7921569 , 0.37254903,
       0.39215687, 0.79607844, 0.3647059 , 0.40392157, 0.8       ,
       0.36078432, 0.4117647 , 0.8       , 0.35686275, 0.41960785,
       0.8039216 , 0.34901962, 0.42745098, 0.80784315, 0.34509805,
       0.4392157 , 0.80784315, 0.3372549 , 0.44705883, 0.8117647 ,
       0.33333334, 0.45490196, 0.8156863 , 0.32941177, 0.46666667,
       0.8156863 , 0.32156864, 0.4745098 , 0.81960785, 0.31764707,
       0.4862745 , 0.8235294 , 0.30980393, 0.49411765, 0.8235294 ,
       0.30588236, 0.5058824 , 0.827451  , 0.29803923, 0.5137255 ,
       0.827451  , 0.29411766, 0.5254902 , 0.83137256, 0.28627452,
       0.53333336, 0.8352941 , 0.2784314 , 0.54509807, 0.8352941 ,
       0.27450982, 0.5529412 , 0.8392157 , 0.26666668, 0.5647059 ,
       0.8392157 , 0.2627451 , 0.57254905, 0.84313726, 0.25490198,
       0.58431375, 0.84313726, 0.24705882, 0.5921569 , 0.84705883,
       0.24313726, 0.6039216 , 0.84705883, 0.23529412, 0.6156863 ,
       0.8509804 , 0.22745098, 0.62352943, 0.8509804 , 0.21960784,
       0.63529414, 0.85490197, 0.21568628, 0.64705884, 0.85490197,
       0.20784314, 0.654902  , 0.85882354, 0.2       , 0.6666667 ,
       0.85882354, 0.19607843, 0.6784314 , 0.8627451 , 0.1882353 ,
       0.6862745 , 0.8627451 , 0.18039216, 0.69803923, 0.8666667 ,
       0.17254902, 0.70980394, 0.8666667 , 0.16862746, 0.7176471 ,
       0.8666667 , 0.16078432, 0.7294118 , 0.87058824, 0.15294118,
       0.7411765 , 0.87058824, 0.14901961, 0.7490196 , 0.8745098 ,
       0.14117648, 0.7607843 , 0.8745098 , 0.13333334, 0.77254903,
       0.8745098 , 0.12941177, 0.78039217, 0.8784314 , 0.12156863,
       0.7921569 , 0.8784314 , 0.11764706, 0.8039216 , 0.8784314 ,
       0.11372549, 0.8117647 , 0.88235295, 0.10980392, 0.8235294 ,
       0.88235295, 0.10588235, 0.83137256, 0.88235295, 0.10196079,
       0.84313726, 0.8862745 , 0.09803922, 0.85490197, 0.8862745 ,
       0.09411765, 0.8627451 , 0.8862745 , 0.09411765, 0.8745098 ,
       0.8901961 , 0.09411765, 0.88235295, 0.8901961 , 0.09411765,
       0.89411765, 0.8901961 , 0.09411765, 0.90588236, 0.89411765,
       0.09803922, 0.9137255 , 0.89411765, 0.09803922, 0.9254902 ,
       0.89411765, 0.10196079, 0.93333334, 0.8980392 , 0.10588235,
       0.94509804, 0.8980392 , 0.10980392, 0.9529412 , 0.8980392 ,
       0.11764706, 0.9647059 , 0.9019608 , 0.12156863, 0.972549  ,
       0.9019608 , 0.12941177, 0.98039216, 0.9019608 , 0.13333334,
       0.99215686, 0.90588236, 0.14117648];

function applyColorMap(x) {
    tf.util.assert(
        x.rank === 4, `Expected rank-4 tensor input, got rank ${x.rank}`);
    tf.util.assert(
        x.shape[0] === 1,
        `Expected exactly one example, but got ${x.shape[0]} examples`);
    tf.util.assert(
        x.shape[3] === 1,
        `Expected exactly one channel, but got ${x.shape[3]} channels`);

    return tf.tidy(() => {
        // Get normalized x.
        const EPSILON = 1e-5;
        const xRange = x.max().sub(x.min());
        const xNorm = x.sub(x.min()).div(xRange.add(EPSILON));
        const xNormData = xNorm.dataSync();

        const h = x.shape[1];
        const w = x.shape[2];
        const buffer = tf.buffer([1, h, w, 3]);

        const colorMapSize = RGB_COLORMAP.length / 3;
        for (let i = 0; i < h; ++i) {
        for (let j = 0; j < w; ++j) {
            const pixelValue = xNormData[i * w + j];
            const row = Math.floor(pixelValue * colorMapSize);
            buffer.set(RGB_COLORMAP[3 * row], 0, i, j, 0);
            buffer.set(RGB_COLORMAP[3 * row + 1], 0, i, j, 1);
            buffer.set(RGB_COLORMAP[3 * row + 2], 0, i, j, 2);
        }
        }
        return buffer.toTensor();
    });
}