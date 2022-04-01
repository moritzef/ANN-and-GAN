import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import ArtificialNeuralNetwork.*;
import GAN.*;

public class Main {
    public static void main(String[] args) throws IOException, InterruptedException, ExecutionException {
/*
        int[] distribution = {4,10,10,1};

        float[] input = {0,1,0};
        float[] example1 = {0,0,0};
        float[] example2 = {0,0,1};
        float[] example3 = {0,1,0};
        float[] example4 = {0,1,1};
        float[] example5 = {1,0,0};
        float[] example6 = {1,0,1};
        float[] example7 = {1,1,0};
        float[] example8 = {1,1,1};

        float[] solution1 = {0};
        float[] solution2 = {0};
        float[] solution3 = {1};
        float[] solution4 = {1};
        float[] solution5 = {0};
        float[] solution6 = {0};
        float[] solution7 = {1};
        float[] solution8 = {1};

        float[][][] f = new float[8][2][];
        f[0][0] = example1;
        f[1][0] = example2;
        f[2][0] = example3;
        f[3][0] = example4;
        f[4][0] = example5;
        f[5][0] = example6;
        f[6][0] = example7;
        f[7][0] = example8;

        f[0][1] = solution1;
        f[1][1] = solution2;
        f[2][1] = solution3;
        f[3][1] = solution4;
        f[4][1] = solution5;
        f[5][1] = solution6;
        f[6][1] = solution7;
        f[7][1] = solution8;



        ANN neural = new ANN(new int[]{3,10,100,100,100,100,1},new String[]{"Tanh","Tanh","Tanh","Tanh","Tanh","Tanh","Tanh"},false);
        Executor e = new Executor();
        e.stochasticGradientDescent(neural,f, (float) 0.01,50000);
        Decider decider = new Decider();
        System.out.println(decider.getStringOutput(neural.calculateANN(input)));
*/

/*

        int[] distribution = {3*50*50,500,100,100,100,100,50,20,10,2};

        ANN neural = new ANN(distribution,false);
        Executor e = new Executor();
        TrainingData train = new TrainingData();
        e.stochasticGradientDescent(neural, train.shuffleData(train.getTrainingData(25)), (float) 0.1,1000);
        Decider decider = new Decider();
        System.out.println(decider.getStringOutput(neural.calculateANN(train.getPicture("PicturesToTry\\cat_01.jpg"))));
        System.out.println(decider.getStringOutput(neural.calculateANN(train.getPicture("PicturesToTry\\dog_01.jpg"))));
        System.out.println(decider.getStringOutput(neural.calculateANN(train.getPicture("PicturesToTry\\DOOOG.jpg"))));
        System.out.println(decider.getStringOutput(neural.calculateANN(train.getPicture("PicturesToTry\\Monkey_selfie.jpg"))));
        System.out.println(decider.getStringOutput(neural.calculateANN(train.getPicture("DATA_SET\\Cats\\1.jpg"))));
*/



        ///*
        GANTrainer g = new GANTrainer(new int[]{30,100,3*50*50},new String[]{"ReLU","ReLU","ReLU"}, new int[]{50*50*3,100,100,100,100,20,2}, new String[]{"ReLU","ReLU","ReLU","ReLU","ReLU","ReLU","ReLU"},false );
        TrainingData train = new TrainingData();
        g.trainGAN(1000,train.getPictureSet(1), (float) 0.6);
        train.createImage("1",50,g.gan.calculateGANOutput(g.gan.getRandomVector(30)));
        train.createImage("2",50,g.gan.calculateGANOutput(g.gan.getRandomVector(30)));
        train.createImage("3",50,g.gan.calculateGANOutput(g.gan.getRandomVector(30)));
        train.createImage("4",50,g.gan.calculateGANOutput(g.gan.getRandomVector(30)));
        train.createImage("5",50,g.gan.calculateGANOutput(g.gan.getRandomVector(30)));
        train.createImage("6",50,g.gan.calculateGANOutput(g.gan.getRandomVector(30)));
        //*/
    }


}
