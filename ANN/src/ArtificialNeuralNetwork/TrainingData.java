package ArtificialNeuralNetwork;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class TrainingData {
    public TrainingData(){}

        public float[][][] getTrainingData(int samplesPerFile) throws IOException {
            String catsPath = "DATA_SET\\Cats\\";
            String dogsPath = "DATA_SET\\Dogs\\";
            float[][][] result = new float[samplesPerFile*2][2][];
            for (int i = 0; i <samplesPerFile; i++) {
                //dogs 0
                result[i][0] = getPicture(dogsPath + (i) + ".jpg");
                result[i][1] = new float[]{1, 0};

            }

            for (int i = samplesPerFile; i <samplesPerFile*2; i++) {
                //cats 1
                result[i][0] = getPicture(catsPath + (i) + ".jpg");
                result[i][1] = new float[]{0, 1};
            }
            return result;
        }

        public float[][] getPictureSet(int sampleSize) throws IOException {
            String catsPath = "DATA_SET\\Cats\\";
            String dogsPath = "DATA_SET\\Dogs\\";
            float[][] result = new float[sampleSize][];
            for (int i = 0; i <sampleSize; i++) {
                result[i] = getPicture(dogsPath + (i+4) + ".jpg");
            }
            return result;
        }

        public float[] getPicture(String path) throws IOException {
            int number = 0;
            Color color;
            BufferedImage img = ImageIO.read(new File(path));
            float[] picture = new float[3*img.getHeight()*img.getWidth()];
            for (int y = 0; y < img.getHeight(); y++) {
                for (int x = 0; x < img.getWidth(); x++) {
                    int pixel = img.getRGB(x,y);
                    color = new Color(pixel, true);
                    picture[number] = color.getRed();
                    number++;
                    picture[number] = color.getGreen();
                    number++;
                    picture[number] = color.getBlue();
                    number++;
                }
            }
            return picture;
        }

        public void createImage(String path, int resolution, float[] inputArray) throws IOException {
            BufferedImage img = new BufferedImage(resolution, resolution, BufferedImage.TYPE_INT_RGB);
            int r,g,b;
            int number=0;
            Color color;
            for (int y = 0; y < img.getHeight(); y++) {
                for (int x = 0; x < img.getWidth(); x++) {
                    r = (inputArray[number]>255)? (int)255:((inputArray[number]<0)?(int)0:(int)inputArray[number]);
                    number++;
                    g = (inputArray[number]>255)? (int)255:((inputArray[number]<0)?(int)0:(int)inputArray[number]);
                    number++;
                    b = (inputArray[number]>255)? (int)255:((inputArray[number]<0)?(int)0:(int)inputArray[number]);
                    number++;
                    color = new Color(r,g,b);
                    img.setRGB(x,y,color.getRGB());
                }
            }
            File file = new File(path+ ".jpg");
            ImageIO.write(img, "jpg", file);
        }

        public float[][][] shuffleData(float[][][] trainingData){

            int examples = trainingData.length;
            ArrayList<Integer> random = new ArrayList<Integer>();
            for (int i=0; i<examples; i++) {
                random.add(i);
            }
            Collections.shuffle(random);

            float[][][] result = new float[examples][][];
            for (int i = 0; i < examples; i++) {
                result[i] = trainingData[random.get(i)];
            }
            return result;
        }

        public float[][][] getSubset(float[][][] trainingData, int number){
            float[][][] result = new float[number][][];
            for (int i = 0; i < number; i++) {
                result[i] = trainingData[i];
            }
            return result;
        }

}
