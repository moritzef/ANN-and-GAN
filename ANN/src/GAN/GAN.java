package GAN;

import ArtificialNeuralNetwork.*;

import java.io.IOException;

public class GAN {
    public ANN generator;
    public ANN discriminator;

    public GAN(int[] distributionGenerator, String[] nodeTypesGenerator, int[] distributionDiscriminator, String[] nodeTypesDiscriminator, boolean getFromFile) throws IOException {
        if(!getFromFile) {
            generator = new ANN(distributionGenerator, nodeTypesGenerator, false);
            discriminator = new ANN(distributionDiscriminator, nodeTypesDiscriminator, false);
        }
    }

    public float[] calculateGANOutput(float[] input){
        return generator.calculateANN(input);
    }


    public void trainDiscriminator(float[] trainingData, float alpha){
        float[][] disciminatorRealTraining = new float[2][];
        disciminatorRealTraining[0] = trainingData;
        disciminatorRealTraining[1] = new float[]{1,0};
        discriminator.backpropagation(disciminatorRealTraining,alpha);



        float[][] disciminatorFakeTraining = new float[2][];
        disciminatorFakeTraining[0] = calculateGANOutput(getRandomVector(30));
        disciminatorFakeTraining[1] = new float[]{0,1};
        discriminator.backpropagation(disciminatorFakeTraining,alpha);

    }


    public void trainGenerator(float alpha) {

        float[][] disciminatorFakeTraining = new float[2][];
        disciminatorFakeTraining[0] = calculateGANOutput(getRandomVector(30));
        disciminatorFakeTraining[1] = new float[]{0,1};
        discriminator.backpropagateDeltas(disciminatorFakeTraining,alpha);

        discriminator.calculateDeltaZero(generator.layers[generator.layers.length-1]);

        float[] delta = new float[generator.layers[generator.layers.length-1].nodes.length];
        for (int i = 0; i < delta.length; i++) {
            delta[i] = discriminator.layers[0].nodes[i].delta;
        }

        float[] calculatedPreviousOutput;
        float calculatedDerivation = 0;
        float errorSum;

        for (int j = generator.distribution.length-1; j > 0; j--) {
            calculatedPreviousOutput = generator.layers[j-1].getOutputArray();
            for (int k = 0; k < generator.distribution[j]; k++) {
                calculatedDerivation = generator.layers[j].nodes[k].calculateOutputFromDerivation();
                if(j==generator.distribution.length-1) {
                    this.generator.layers[j].nodes[k].delta = delta[k];
                }else {
                    errorSum = 0;
                    for (int n = 0; n < generator.distribution[j+1]; n++) {
                        errorSum += this.generator.weights[j+1][n][k]*this.generator.layers[j+1].nodes[n].delta;
                    }
                    this.generator.layers[j].nodes[k].delta = errorSum * calculatedDerivation;
                }

                for (int l = 0; l < generator.distribution[j - 1]; l++) {
                    this.generator.weights[j][k][l] += alpha * calculatedPreviousOutput[l] * this.generator.layers[j].nodes[k].delta;
                }
                this.generator.bias[j][k] += alpha * this.generator.layers[j].nodes[k].delta;
            }
        }
        generator.updateConfig();
    }

    public float[] getRandomVector(int size){
        float[] vector = new float[size];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float)  (-1+(2*((float) Math.random())));
        }
        return vector;
    }
}
