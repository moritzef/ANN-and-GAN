package ArtificialNeuralNetwork.Node;

public abstract class Node {
    public Node[] inputs;
    public float[] weights;
    public float output;
    public float bias;
    public float delta;

    public Node(Node[] inputs, float[] weights, float bias){
        this.inputs = inputs;
        this.weights = weights;
        this.bias = bias;
    }

    public void updateNode(float[] weights, float bias){
        this.weights = weights;
        this.bias = bias;
    }

    public void calculateOutput(){
            this.output = function();
    }

    public float calculateInput(){
        float result = 0;
        for(int i = 0; i<inputs.length; i++){
            result = result + (inputs[i].output * weights[i]);
        }
        return result+bias;
    }

    public void setOutput(float output){
        this.output = output;
    }

    public abstract float calculateOutputFromDerivation();
    public abstract float function();
}
