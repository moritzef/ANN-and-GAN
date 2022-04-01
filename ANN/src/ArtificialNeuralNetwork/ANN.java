package ArtificialNeuralNetwork;

import ArtificialNeuralNetwork.Node.*;
import java.io.*;
import java.util.Arrays;
import java.util.Scanner;



public class ANN {
    public Layer[] layers;
    public float[][][] weights;
    public float[][] bias;
    public int[] distribution;

    public ANN(int[] distribution, String[] nodeTypes, boolean fromExistingFile) throws IOException {
        this.distribution = distribution;
        if (fromExistingFile) {
            getCurrentConfigFromFile();
        } else {
            createNewConfig(distribution);
        }
        layers = new Layer[distribution.length];
        layers[0] = new Layer(distribution[0], nodeTypes[0], null, null,null,true);
        for (int i = 1; i < distribution.length; i++) {
            layers[i] = new Layer(distribution[i],nodeTypes[i], layers[i-1].nodes, weights[i],bias[i],false);
        }
    }

    public float[] calculateANNWithoutInput(){
        for (int i = 0; i < layers.length; i++) {
            layers[i].calculateLayerOutput();
        }
        Node[] lastNodes = layers[layers.length-1].nodes;
        float[] result = new float[lastNodes.length];
        for (int i = 0; i < lastNodes.length; i++) {
            result[i] = lastNodes[i].output;
        }
        return result;
    }

    public float[] calculateANN(float[] input){
        for (int i = 0; i < input.length; i++) {
            layers[0].nodes[i].setOutput(input[i]);
        }
        for (int i = 1; i < layers.length; i++) {
            layers[i].calculateLayerOutput();
        }
        Node[] lastNodes = layers[layers.length-1].nodes;
        float[] result = new float[lastNodes.length];
        for (int i = 0; i < lastNodes.length; i++) {
            result[i] = lastNodes[i].output;
        }
        return result;
    }

    public void createNewConfig(int[] distribution) throws IOException {
        this.weights = new float[distribution.length][][];
        float[][] currWeight;
        for (int i = 1; i < distribution.length; i++) {
            currWeight = new float[distribution[i]][distribution[i-1]];
            for (int j = 0; j < distribution[i] ; j++) {
                for (int k = 0; k < distribution[i-1]; k++) {
                    currWeight[j][k] = (-1+(2*((float) Math.random())))*0.001f;
                }
            }
            this.weights[i] = currWeight;
        }

        this.bias = new float[distribution.length][];
        float[] currBias;
        for (int i = 1; i < distribution.length; i++) {
            currBias = new float[distribution[i]];
            for (int j = 0; j < distribution[i] ; j++) {
                //currBias[j] = (-1+(2*((float) Math.random())));
                currBias[j] = -0.1f*((float) Math.random());
            }
            this.bias[i] = currBias;
        }
        safeAsFile();
    }

    public void safeAsFile() throws IOException {
        File file1 = new File("Config.txt");
        BufferedWriter writer = new BufferedWriter(new FileWriter(file1));
        for (int i = 1; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    if(weights[i][j][k]!=0) {
                        writer.write(i + ":" + j + "$" + k + "=" + weights[i][j][k]);
                        writer.newLine();
                    }
                }
            }
        }
        writer.write("next");
        writer.newLine();
        for (int i = 1; i < bias.length; i++) {
            for (int j = 0; j < bias[i].length; j++) {
                if(bias[i][j]!=0) {
                    writer.write(i + ":" + j + "=" + bias[i][j]);
                    writer.newLine();
                }
            }
        }

        writer.close();

    }

    public void getCurrentConfigFromFile() throws FileNotFoundException {
        InputStream ins = new FileInputStream("Config.txt");
        Scanner obj = new Scanner(ins);
        int i,j,k;
        float w,b;
        boolean next = false;
        String current = "";
        while (obj.hasNextLine()){
            current = obj.nextLine();
            if(current.equals("next")) {
                next = true;
                continue;
            }
            if(!next) {
                i = Integer.parseInt(current.substring(0, current.indexOf(":")));
                j = Integer.parseInt(current.substring(current.indexOf(":") + 1, current.indexOf("$")));
                k = Integer.parseInt(current.substring(current.indexOf("$") + 1, current.indexOf("=")));
                w = Float.parseFloat(current.substring(current.indexOf("=") + 1));
                this.weights[i][j][k] = w;
            }else {
                i = Integer.parseInt(current.substring(0, current.indexOf(":")));
                j = Integer.parseInt(current.substring(current.indexOf(":") + 1, current.indexOf("=")));
                b = Float.parseFloat(current.substring(current.indexOf("=") + 1));
                this.bias[i][j] = b;
            }


        }


    }

    public void updateConfig() {
        for (int i = 1; i < layers.length; i++) {
            layers[i].updateLayer(weights[i],bias[i]);
        }
    }

    public void calculateDeltaZero(Layer layer){
        float errorSum;
        for (int i = 0; i < layers[0].nodes.length; i++) {
            errorSum = 0;
            for (int n = 0; n < distribution[1]; n++) {
                errorSum += this.weights[1][n][i]*this.layers[1].nodes[n].delta;
            }
            this.layers[0].nodes[i].delta = errorSum * layer.nodes[i].calculateOutputFromDerivation();
        }
    }

    public void backpropagation(float[][] trainingData, float alpha) {
        float[] calculatedOutput, calculatedPreviousOutput;
        float calculatedDerivation = 0;
        float errorSum, currentLoss;
        float totalLoss = 0;
            calculatedOutput = calculateANN(trainingData[0]);
            if(calculatedOutput[0]>2)
                System.out.println("jetzt");
            for (int j = distribution.length-1; j > 0; j--) {
                calculatedPreviousOutput = layers[j-1].getOutputArray();
                for (int k = 0; k < distribution[j]; k++) {
                    calculatedDerivation = layers[j].nodes[k].calculateOutputFromDerivation();
                    if(j==distribution.length-1) {
                        currentLoss = trainingData[1][k] - calculatedOutput[k];
                        totalLoss += Math.abs(currentLoss);
                        this.layers[j].nodes[k].delta = currentLoss * calculatedDerivation;
                    }else {
                        errorSum = 0;
                        for (int n = 0; n < distribution[j+1]; n++) {
                            errorSum += this.weights[j+1][n][k]*this.layers[j+1].nodes[n].delta;
                        }
                        this.layers[j].nodes[k].delta = errorSum * calculatedDerivation;
                    }

                    for (int l = 0; l < distribution[j - 1]; l++) {
                            this.weights[j][k][l] += alpha * calculatedPreviousOutput[l] * this.layers[j].nodes[k].delta;
                    }
                    this.bias[j][k] += alpha * this.layers[j].nodes[k].delta;
                }
            }
            updateConfig();
            System.out.println("current Loss in example "  +" is: "+ totalLoss);
    }

    public void backpropagateDeltas(float[][] trainingData, float alpha) {
        float[] calculatedOutput, calculatedPreviousOutput;
        float calculatedDerivation = 0;
        float errorSum, currentLoss;
        float totalLoss = 0;
        calculatedOutput = calculateANN(trainingData[0]);

        float[][][] newWeights = copyWeights();

        for (int j = distribution.length-1; j > 0; j--) {
            calculatedPreviousOutput = layers[j-1].getOutputArray();
            for (int k = 0; k < distribution[j]; k++) {
                calculatedDerivation = layers[j].nodes[k].calculateOutputFromDerivation();
                if(j==distribution.length-1) {
                    currentLoss = trainingData[1][k] - calculatedOutput[k];
                    totalLoss += Math.abs(currentLoss);
                    this.layers[j].nodes[k].delta = currentLoss * calculatedDerivation;
                }else {
                    errorSum = 0;
                    for (int n = 0; n < distribution[j+1]; n++) {
                        errorSum += newWeights[j+1][n][k]*this.layers[j+1].nodes[n].delta;
                    }
                    this.layers[j].nodes[k].delta = errorSum * calculatedDerivation;
                }

                for (int l = 0; l < distribution[j - 1]; l++) {
                    newWeights[j][k][l] += alpha * calculatedPreviousOutput[l] * this.layers[j].nodes[k].delta;
                }
            }
        }

        System.out.println("current Deltas Loss is" + totalLoss);
    }

    public float getLossOfCurrentExample(float[] loss){
        float currentLoss =0;
        for (float element:loss) {
            currentLoss+= Math.abs(element);
        }
        return  currentLoss;
    }

    public float getTotalLossOfCurrentExample(float[][] exampleInputs, float[][] exampleOutputs){
        float totalLoss = 0;
        for (int i = 0; i < exampleInputs.length; i++) {
            for (int j = 0; j < exampleInputs[i].length; j++) {
                totalLoss += Math.abs(exampleOutputs[i][j] - calculateANN(exampleInputs[i])[j]);
            }
        }
        return totalLoss;
    }
    public float[][][] getWeights(){
        return this.weights;
    }

    public float[][] getBias(){
        return this.bias;
    }

    public float[][][] copyWeights(){
        float[][][] result = new float[this.weights.length][][];
        float[][] partialResult;
        float[] singlePartialResult;
        for (int i = 1; i < this.weights.length; i++) {
            partialResult = new float[this.weights[i].length][];
            for (int j = 0; j < this.weights[i].length; j++) {
                singlePartialResult = new float[this.weights[i][j].length];
                for (int k = 0; k < this.weights[i][j].length; k++) {
                    singlePartialResult[k] = this.weights[i][j][k];
                }
                partialResult[j] = singlePartialResult;
            }
            result[i] = partialResult;
        }
        return result;
    }
}

