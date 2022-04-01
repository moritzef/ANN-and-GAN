package ArtificialNeuralNetwork;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class ThreadEnhanced {
    public ThreadEnhanced(){}

    public ANN executeInThreads(ANN ann, String[] nodeTypes, float[][][] trainingData, float alpha) throws InterruptedException, ExecutionException, IOException {
        ArrayList<ANNThread> tasks = new ArrayList<>();
        for (int i = 0; i < trainingData.length; i++) {
            tasks.add(new ANNThread(ann, nodeTypes,trainingData[i],alpha));
        }
        ExecutorService es = Executors.newCachedThreadPool();
        List<Future<struct>> futures = es.invokeAll(tasks);
        es.shutdown();
        ArrayList<struct> structs = new ArrayList<>();
        for (Future<struct> future : futures) {
            structs.add(future.get());
        }
        List<float[][][]> weights = structs.stream().map(struct::getWeights).collect(Collectors.toList());
        List<float[][]> bias = structs.stream().map(struct::getBias).collect(Collectors.toList());

        ann.weights = getAverageWeights(weights.toArray(new float[][][][]{}));
        ann.bias = getAverageBias(bias.toArray(new float[][][]{}));
        return ann;
    }

    public float[][][] getAverageWeights(float[][][][] inputs){
        float[][][] result = inputs[0];
        float sum;
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                for (int k = 0; k < result[i][j].length; k++) {

                    sum = 0;
                    for (int l = 0; l < inputs.length; l++) {
                        sum += inputs[l][i][j][k];
                    }
                    result[i][j][k] = sum/inputs.length;

                }
            }
        }
        return result;
    }

    public float[][] getAverageBias(float[][][] inputs){
        float[][] result = inputs[0];
        float sum = 0;
            for (int j = 0; j < result.length; j++) {
                for (int k = 0; k < result[j].length; k++) {
                    sum = 0;
                    for (int l = 0; l < inputs.length; l++) {
                        sum += inputs[l][j][k];
                    }
                    result[j][k] = sum / inputs.length;
                }
            }

        return result;
    }

    public class ANNThread implements Callable<struct> {
        ANN ann;
        float[][] trainingData;
        float alpha;

        public ANNThread(ANN ann, String[] nodeType, float[][] trainingData, float alpha) throws IOException {
            this.ann = new ANN(ann.distribution, nodeType,true);
            this.ann.weights = ann.weights;
            this.ann.bias = ann.bias;
            this.ann.layers = ann.layers;
            this.trainingData = trainingData;
            this.alpha = alpha;
        }

        @Override
        public struct call() throws IOException {
            struct s = new struct(new float[][][]{},new float[][]{});
            ann.backpropagation(trainingData,alpha);
            s.setWeights(ann.getWeights());
            s.setBias(ann.getBias());
            return s;
        }
    }

    public static class struct{
        public float[][][] weights;
        public float[][] bias;


        public struct(float[][][] weights, float[][] bias) {
            this.weights = weights;
            this.bias = bias;
        }

        public float[][][] getWeights() {
            return weights;
        }

        public float[][] getBias() {
            return bias;
        }

        public void setWeights(float[][][] weights) {
            this.weights = weights;
        }

        public void setBias(float[][] bias) {
            this.bias = bias;
        }
    }
}
