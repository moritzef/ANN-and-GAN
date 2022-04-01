package ArtificialNeuralNetwork.Node;

public class ReLUNode extends Node{
    public ReLUNode(Node[] inputs, float[] weights, float bias) {
        super(inputs, weights, bias);
    }

    @Override
    public float function(){
        float inp = calculateInput();
        return (inp>0)?inp:0.001f*inp;
    }

    @Override
    public float calculateOutputFromDerivation(){
        float inp = calculateInput();
        return  (inp>0)?1:0.001f;
    }
}
