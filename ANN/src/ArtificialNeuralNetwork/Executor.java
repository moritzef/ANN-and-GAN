package ArtificialNeuralNetwork;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

public class Executor {
    TrainingData trainer;
    public Executor(){
        this.trainer = new TrainingData();
    }

    public ANN miniBatchGradientDescent(ANN ann, String[] nodeTypes, float[][][] trainingData, float alpha, int loops) throws InterruptedException, IOException, ExecutionException {
        ThreadEnhanced te = new ThreadEnhanced();
        for (int i = 0; i < loops; i++) {
            ann = te.executeInThreads(ann,nodeTypes,trainer.getSubset(trainer.shuffleData(trainingData),32)  ,alpha);
            System.out.println("round"+i);
        }
        ann.safeAsFile();
        return ann;
    }

    public ANN batchGradientDescent(ANN ann, String[] nodeTypes, float[][][] trainingData, float alpha, int loops) throws InterruptedException, ExecutionException, IOException {
        ThreadEnhanced te = new ThreadEnhanced();
        for (int i = 0; i < loops; i++) {
            ann = te.executeInThreads(ann, nodeTypes, trainingData, alpha);
            System.out.println("round"+i);
        }
        ann.safeAsFile();
        return ann;
    }

    public void stochasticGradientDescent(ANN neural, float[][][] trainingData,float alpha, int loops) throws IOException {
        for (int i = 0; i < loops; i++) {
            for (int j = 0; j < trainingData.length; j++) {
                System.out.println("round " + i + " example " + j);
                neural.backpropagation(trainingData[j],alpha);
            }
        }
        neural.safeAsFile();
    }
}
