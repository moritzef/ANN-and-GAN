package ArtificialNeuralNetwork;

import java.util.Arrays;

public class Decider {
    public Decider(){}

    public String getStringOutput(float[] output){
        float maxProbability = -1;
        int index = -1;
        for (int i = 0; i < output.length; i++) {
            System.out.println("output "+output[i]);
            if(output[i]>maxProbability){
                maxProbability = output[i];
                index = i;
            }
        }
        return "The Result is " + index + " with a probability of " + maxProbability + "%" ;
    }
}
