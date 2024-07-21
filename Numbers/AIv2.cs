using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;

namespace Numbers
{
    internal class AIv2
    {
        private List<Matrix<double>> Layers;
        private List<Vector<double>> Bias;
        private List<Matrix<double>> Weight;
        private double loss = 0;

        private int[] SizeEachLayer { get; }
        private int CountAllLayers { get; }

        private int BACH_SIZE = 50;
        private int COUNT_EPOCHS = 100;
        private double LearnSpeed = 0.001;
        private double MaxIn = 255;
        private double MinIn = 0;

        private byte[] buffL;
        private byte[][] buffI;

        public AIv2(int[] sizeEachLayer)
        {
            SizeEachLayer = sizeEachLayer;
            CountAllLayers = sizeEachLayer.Length;

            CreatePerceptron();
            ReadMnistTrainingFile();
        }

        private void CreatePerceptron()
        {
            Layers = new List<Matrix<double>>(CountAllLayers);
            Weight = new List<Matrix<double>>(CountAllLayers - 1);
            Bias = new List<Vector<double>>(CountAllLayers - 1);

            ContinuousUniform ran = new ContinuousUniform(-1 * Math.Sqrt(2.0 / (SizeEachLayer[0] + SizeEachLayer[CountAllLayers - 1])), Math.Sqrt(2.0 / (SizeEachLayer[0] + SizeEachLayer[CountAllLayers - 1])));

            for (int i = 0; i < CountAllLayers; i++)
            {
                Layers.Add(Matrix<double>.Build.Dense(BACH_SIZE, SizeEachLayer[i]));
                if (i != CountAllLayers - 1)
                {
                    Weight.Add(Matrix<double>.Build.Random(SizeEachLayer[i], SizeEachLayer[i + 1], ran));
                    Bias.Add(Vector<double>.Build.Dense(SizeEachLayer[i + 1]));
                }
            }
        }

        public void Training()
        {
            for (int i = 0; i < COUNT_EPOCHS; i++)
            {
                int c = 0;
                Shuffle();
                for (int j = 0; j < buffL.Length / BACH_SIZE; j++)
                {
                    Matrix<double> bachImage = Matrix<double>.Build.Dense(BACH_SIZE, 28 * 28);
                    Vector<double> ans = Vector<double>.Build.Dense(BACH_SIZE);
                    for (int k = 0; k < BACH_SIZE; k++)
                    {
                        ans[k] = buffL[k + c];
                        for (int l = 0; l < 28 * 28; l++)
                            bachImage[k, l] = buffI[k + c][l];
                    }

                    BTrain(bachImage, ans);
                    c += BACH_SIZE;
                    Console.WriteLine(loss);
                    break;
                }
                break;
            }
        }
        public void BTrain(Matrix<double> inputLayer, Vector<double> answer)
        {
            for (int i = 0; i < CountAllLayers - 1; i++)
            {
                if (i == 0)
                    Layers[i + 1] = Normalize(inputLayer) * Weight[i];
                else
                    Layers[i + 1] = Layers[i] * Weight[i];

                for (int j = 0; j < Layers[i].RowCount; j++)
                    Layers[i + 1].SetRow(j, Layers[i + 1].Row(j) + Bias[i]);

                if (i != CountAllLayers - 1)
                    Layers[i + 1] = Relu(Layers[i + 1]);
                else
                    Layers[i + 1] = SoftMax(Layers[i + 1]);
            }

            Matrix<double> ans = Matrix<double>.Build.Dense(BACH_SIZE, SizeEachLayer[CountAllLayers - 1]);
            for (int i = 0; i < answer.Count; i++)
                ans[i, Convert.ToInt32(answer[i])] = 1.0;

            loss = Error(Layers[CountAllLayers - 1], ans);

            Matrix<double> dEdt;
            Matrix<double> dEdH = Matrix<double>.Build.Dense(BACH_SIZE, SizeEachLayer[CountAllLayers - 2]);

            for (int i = CountAllLayers - 1; i > 0; i--)
            {
                if (i == CountAllLayers - 1)
                {
                    dEdH = 2 * (Layers[i] - ans);
                    dEdt = dEdH.PointwiseMultiply(DetSoftMax(Layers[i]));
                }
                else
                    dEdt = dEdH.PointwiseMultiply(DetRelu(Layers[i]));

                Weight[i - 1] -= Layers[i - 1].Transpose() * dEdt * LearnSpeed;
                Bias[i - 1] -= dEdt.ColumnSums() * LearnSpeed;
                dEdH = dEdt * Weight[i - 1].Transpose();
            }

        }

        public double[] Predict(double[] inputLayer)
        {
            Matrix<double> vector = Normalize(inputLayer);
            List<Matrix<double>> Layer = new List<Matrix<double>>(CountAllLayers);

            for (int i = 0; i < CountAllLayers; i++)
            {
                if (i == 0)
                    Layer.Add(Matrix<double>.Build.DenseOfRowArrays(inputLayer));
                else
                    Layer.Add(Matrix<double>.Build.Dense(1, SizeEachLayer[i]));
            }

            for (int i = 0; i < CountAllLayers - 1; i++)
            {
                if (i == 0)
                    Layer[i + 1] = vector * Weight[i];
                else
                    Layer[i + 1] = Layer[i] * Weight[i];

                
                Layer[i + 1] += Bias[i].ToRowMatrix();

                if (i != CountAllLayers - 2)
                    Layer[i + 1] = Relu(Layer[i + 1]);
                else
                    Layer[i + 1] = SoftMax(Layer[i + 1]);
            }
            return Layer[CountAllLayers - 1].Row(0).ToArray();
        }

        private double Error(Matrix<double> layer, Matrix<double> ans)
        {
            double rez = 0;

            for (int i = 0; i < BACH_SIZE; i++)
                for (int j = 0; j < layer.ColumnCount; j++)
                    rez += Math.Pow(layer[i, j] - ans[i, j], 2);

            return rez / BACH_SIZE;
        }

        private Matrix<double> Relu(Matrix<double> mat)
        {
            Matrix<double> matrix = Matrix<double>.Build.Dense(mat.RowCount, mat.ColumnCount);
            for (int i = 0; i < mat.RowCount; i++)
                for (int j = 0; j < mat.ColumnCount; j++)
                {
                    if (mat[i, j] > 0)
                        matrix[i, j] = mat[i, j];
                    else
                        matrix[i, j] = 0;
                }
            return matrix;
        }
        private Matrix<double> DetRelu(Matrix<double> mat)
        {
            Matrix<double> ans = Matrix<double>.Build.Dense(mat.RowCount, mat.ColumnCount);

            for (int i = 0; i < mat.RowCount; i++)
                for (int j = 0; j < mat.ColumnCount; j++)
                    ans[i, j] = mat[i,j] > 0 ? 1 : 0;

            return ans;
        }

        private Matrix<double> SoftMax(Matrix<double> mat)
        {
            Matrix<double> ans = Matrix<double>.Build.Dense(mat.RowCount, mat.ColumnCount);

            for (int i = 0; i < mat.RowCount; i++)
            {
                double sum = 0;

                for (int j = 0; j < mat.ColumnCount; j++)
                    sum += Math.Exp(mat[i, j]);
                for (int j = 0; j < mat.ColumnCount; j++)
                    ans[i, j] = Math.Exp(mat[i, j]) / sum;
            }
            return ans;
        }
        private Matrix<double> DetSoftMax(Matrix<double> mat)
        {
            Matrix<double> ans = SoftMax(mat);

            for (int i = 0; i < mat.RowCount; i++)
                for (int j = 0; j < mat.ColumnCount; j++)
                    ans[i, j] *= 1 - ans[i, j];

            return ans;
        }

        private Matrix<double> Normalize(double[] vec)
        {
            Matrix<double> ans = Matrix<double>.Build.DenseOfRowArrays(vec);

            for (int i = 0; i < vec.Length; i++)
                ans[0, i] = (ans[0, i] - MinIn) / (MaxIn - MinIn);

            return ans;
        }
        private Matrix<double> Normalize(Matrix<double> mat)
        {
            Matrix<double> ans = Matrix<double>.Build.DenseOfMatrix(mat);

            for (int i = 0; i < ans.RowCount; i++)
                for (int j = 0; j < ans.ColumnCount; j++)
                    ans[i, j] = (ans[i, j] - MinIn) / (MaxIn - MinIn);

            return ans;
        }
        private void ReadMnistTrainingFile()
        {

            FileStream trainL = new FileStream(@"C:\Users\nikit\Работы\C#\Numbers\Numbers\data\train-labels.idx1-ubyte", FileMode.Open);
            FileStream trainI = new FileStream(@"C:\Users\nikit\Работы\C#\Numbers\Numbers\data\train-images.idx3-ubyte", FileMode.Open);

            BinaryReader brL = new BinaryReader(trainL);
            BinaryReader brI = new BinaryReader(trainI);

            buffL = new byte[trainL.Length - 8];
            buffI = new byte[(trainI.Length - 16) / (28 * 28)][];

            buffI[0] = new byte[28 * 28];

            brL.Read(buffL, 0, 8);
            brI.Read(buffI[0], 0, 16);

            for (int i = 0; i < buffL.Length; i++)
                buffL[i] = brL.ReadByte();

            for (int i = 0; i < buffI.Length; i++)
            {
                buffI[i] = new byte[28 * 28];

                for (int j = 0; j < buffI[i].Length; j++)
                    buffI[i][j] = brI.ReadByte();
            }
        }
        private void Shuffle()
        {
            Random r = new Random();
            for (int i = buffL.Length - 1; i >= 1; i--)
            {
                int j = r.Next(i + 1);
                var temp = buffI[j];
                buffI[j] = buffI[i];
                buffI[i] = temp;
                var temp2 = buffL[j];
                buffL[j] = buffL[i];
                buffL[i] = temp2;
            }
        }
    }
}
