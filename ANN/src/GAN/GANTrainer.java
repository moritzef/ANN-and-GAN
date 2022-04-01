package GAN;

import ArtificialNeuralNetwork.Decider;

import java.io.IOException;
import java.util.Arrays;

public class GANTrainer {
    public GAN gan;
    public GANTrainer(int[] distributionGenerator, String[] nodeTypesGenerator, int[] distributionDiscriminator, String[] nodeTypesDiscriminator, boolean getFromFile) throws IOException {
        gan = new GAN(distributionGenerator, nodeTypesGenerator, distributionDiscriminator, nodeTypesDiscriminator, getFromFile);
    }

    public void trainGAN(int loops, float[][] trainingData,float alpha) {
        Decider d = new Decider();
        for (int i = 0; i < loops; i++) {
            for (int k = 0; k < trainingData.length*40; k++) {
                System.out.println("discriminator loop " + i + " example " +k +":");
                gan.trainDiscriminator(trainingData[0], alpha);
            }

            for (int j = 0; j < trainingData.length*40; j++) {
                System.out.println("generator loop " + i + " example " + j +":");
                gan.trainGenerator(alpha);

            }
        }
    }
}
