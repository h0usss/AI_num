using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Reflection.Emit;
using MathNet.Numerics.LinearAlgebra;

namespace Numbers
{
    internal class AI
    {
        private double[][]   Layers { get; set; }
        private double[][]   Bias { get; set; }
        private double[][,] Weight { get; set; }
        
        private int[] SizeEachLayer { get; }
        private int   CountAllLayers { get; }

        private int BACH_SIZE = 500;
        private int COUNT_EPOCHS = 1;
        private double LearnSpeed = 0.002;
        private double MaxIn = 255;
        private double MinIn = 0;

        private byte[] buffL;
        private byte[][] buffI;

        public AI(int[] sizeEachLayer) 
        {
            SizeEachLayer = sizeEachLayer;
            CountAllLayers = sizeEachLayer.Length;
            
            CreatePerceptron();
            ReadMnistTrainingFile();
        }

        private void CreatePerceptron()
        {
            Random random = new Random();
            
            Layers = new double[CountAllLayers][];
            Weight = new double[CountAllLayers - 1][,];
            Bias = new double[CountAllLayers - 1][];

            for (int i = 0; i < CountAllLayers; i++)
            {
                Layers[i] = new double[SizeEachLayer[i]];

                for (int j = 0; j < SizeEachLayer[i]; j++)                  
                    Layers[i][j] = 0;
            }

            for (int i = 0; i < CountAllLayers - 1; i++)
            {
                Weight[i] = new double[SizeEachLayer[i], SizeEachLayer[i + 1]];

                for (int j = 0; j < SizeEachLayer[i]; j++)
                    for (int k = 0; k < SizeEachLayer[i + 1]; k++)
                        //Weight[i][j, k] = random.NextDouble() * (4.0 / (SizeEachLayer[0] + SizeEachLayer[SizeEachLayer.Length - 1])) - (2.0 / (SizeEachLayer[0] + SizeEachLayer[SizeEachLayer.Length - 1]));
                        Weight[i][j, k] = (random.NextDouble() - 0.5) * 2.0 * Math.Sqrt(2.0 / (SizeEachLayer[i] + SizeEachLayer[i + 1]));
               
                Bias[i] = new double[SizeEachLayer[i + 1]];
                for (int j = 0; j < SizeEachLayer[i + 1]; j++)
                    //Bias[i][j] = random.Next(-100, 100) / 100.0;
                    //Bias[i][j] = (random.NextDouble() - 0.5) * 2.0 * Math.Sqrt(2.0 / (SizeEachLayer[i] + SizeEachLayer[i + 1]));
                    Bias[i][j] = 0;
            }
        }

        public double[] Predict(double[] inputLayer)
        {
            SetZeroNeurons();
            SetInputLayer(inputLayer);

            for (int i = 0; i < Weight.Length; i++)
            {
                for (int j = 0; j < Weight[i].GetUpperBound(1) + 1; j++)
                {
                    for (int k = 0; k < Weight[i].GetUpperBound(0) + 1; k++)
                    {
                        Layers[i + 1][j] += Layers[i][k] * Weight[i][k,j];
                    }

                    Layers[i + 1][j] += Bias[i][j];
                }

                if (i != Weight.Length - 1)
                    Layers[i + 1] = Relu(Layers[i + 1]);
            }

            return SoftMax(Layers[Layers.Length - 1]);
        }
        public void BTrain(byte[,] inputLayer, int[] answer)
        {
            double[][,] Layers = new double[SizeEachLayer.Length][,];
            for (int i = 0; i < Layers.Length; i++)
            {
                if (i == 0)
                {
                    Layers[i] = Normalize(inputLayer, MinIn, MaxIn);
                }
                else
                {
                    if (i != Layers.Length - 1)
                    {
                        Layers[i] = Relu(Summ(Mult(Layers[i - 1], Weight[i - 1]), Bias[i - 1]));

                        for (int j = 0; j < Layers[i].GetUpperBound(0) + 1; j++)
                            for (int k = 0; k < Layers[i].GetUpperBound(1) + 1; k++)
                                if (double.IsNaN(Layers[i][j, k]))
                                    continue;
                    }
                    else
                    {
                        Layers[i] = SoftMax(Summ(Mult(Layers[i - 1], Weight[i - 1]), Bias[i - 1]));
                    }
                }
            }

            double[,] ansArr = new double[BACH_SIZE, SizeEachLayer[SizeEachLayer.Length - 1]];
            for (int i = 0; i < BACH_SIZE; i++)
                ansArr[i, answer[i]] = 1;

            double[,] dEdt;
            double[,] dEdH = new double[BACH_SIZE, SizeEachLayer[Layers.Length - 2]];

            for (int i = CountAllLayers - 1; i > 0; i--)
            {
                if (i == CountAllLayers - 1)
                {
                    dEdH = Mult(Sub(Layers[i], ansArr), 2);
                    dEdt = MultEl(dEdH, DetSoftMax(Layers[i]));
                }
                else
                    dEdt = MultEl(dEdH, DetRelu(Layers[i]));

                Weight[i - 1] = Sub(Weight[i - 1], Mult(Mult(Trans(Layers[i - 1]), Summ(dEdt)), LearnSpeed));
                Bias[i - 1] = Sub(Bias[i - 1], Mult(Summ(dEdt), LearnSpeed));
                dEdH = Mult(dEdt, Trans(Weight[i - 1]));
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
                    byte[,] bachImage = new byte[BACH_SIZE, 28 * 28];
                    int[] ans = new int[BACH_SIZE];
                    for (int k = 0; k < BACH_SIZE; k++)
                    {
                        ans[k] = buffL[k + c];
                        for (int l = 0; l < 28 * 28; l++)
                            bachImage[k, l] = buffI[k + c][l];
                    }

                    BTrain(bachImage, ans);
                    c += BACH_SIZE;
                }
            }
        }
        private double Error(double[] outputLayer, int answer)
        {
            int[] ansArr = new int[outputLayer.Length];
            double[] outL = new double[outputLayer.Length];

            ansArr[answer] = 1;
            outL = SoftMax(outputLayer);

            for (int i = 0; i < outL.Length; i++)
                outL[i] = Math.Pow(outL[i] - ansArr[i], 2);

            return outL.Sum();
        }





        private double Relu(double weight)
        {
            return Math.Max(0, weight);
        }
        private double[] Relu(double[] weight)
        {
            double[] wth = new double[weight.Length];

            for (int i = 0; i < weight.Length; i++)
                wth[i] = Relu(weight[i]);
            return wth;
        }
        private double[,] Relu(double[,] matrix)
        {
            double[,] wth = new double[matrix.GetUpperBound(0) + 1, matrix.GetUpperBound(1) + 1];

            for (int i = 0; i < matrix.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < matrix.GetUpperBound(1) + 1; j++)
                    wth[i, j] = Relu(matrix[i, j]);
            return wth;
        }
        private double DetRelu(double value)
        {
            if (value <= 0) return 0;
            return 1;
        }
        private double[] DetRelu(double[] value)
        {
            double[] ans = new double[value.Length];

            for (int i = 0; i < value.Length; i++) 
                ans[i] = DetRelu(value[i]);
            return ans;
        }
        private double[,] DetRelu(double[,] matrix)
        {
            double[,] rez = new double[matrix.GetUpperBound(0) + 1, matrix.GetUpperBound(1) + 1];

            for (int i = 0; i < matrix.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < matrix.GetUpperBound(1) + 1; j++)
                    rez[i, j] = DetRelu(matrix[i, j]);
            return rez;
        }

        private double[,] SoftMax(double[,] matrix)
        {
            double sum = 0;
            double[] vec = new double[matrix.GetUpperBound(1) + 1];
            double[,] rez = new double[matrix.GetUpperBound(0) + 1, matrix.GetUpperBound(1) + 1];

            for (int i = 0; i < matrix.GetUpperBound(0) + 1; i++)
            {
                for (int j = 0; j < matrix.GetUpperBound(1) + 1; j++)
                {
                    vec[j] = matrix[i, j];
                    sum += Math.Exp(vec[j]);
                }

                for (int j = 0; j < matrix.GetUpperBound(1) + 1; j++)
                    rez[i, j] = Math.Exp(vec[j]) / sum;
            }
            return rez;
        }
        private double[] SoftMax(double[] vector)
        {
            double sum = 0;
            double[] vec = Normalize(vector);

            for (int i = 0; i < vector.Length; i++)
                sum += Math.Exp(vector[i]);
            for (int i = 0; i < vector.Length; i++)
                vec[i] = Math.Exp(vector[i]) / sum;
            return vec;
        }
        private double DetSoftMax(double[] vector, int I)
        {
            double[] vecSM = SoftMax(vector);
            return vecSM[I] * (1 - vecSM[I]);
        }
        private double[] DetSoftMax(double[] vector)
        {
            double[] vecSM = SoftMax(vector);
            for (int i = 0; i < vecSM.Length; i++)
                vecSM[i] = vecSM[i] * (1 - vecSM[i]);
            return vecSM;
        }
        private double[,] DetSoftMax(double[,] matrix)
        {
            double[,] vecSM = SoftMax(matrix);
            for (int i = 0; i < vecSM.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < vecSM.GetUpperBound(1) + 1; j++)
                    vecSM[i, j] = vecSM[i, j] * (1 - vecSM[i, j]);
            return vecSM;
        }

        private double Normalize(double weight, double MinIn, double MaxIn)
        {
            return (weight - MinIn)/(MaxIn - MinIn);
        }
        private double[] Normalize(double[] vec)
        {
            double[] rez = new double[vec.Length];
            for (int i = 0; i < rez.Length; i++)
                rez[i] = (vec[i] - vec.Min()) / (vec.Max() - vec.Min());
            return rez;
        }
        private double[,] Normalize(byte[,] matrix, double MinIn, double MaxIn)
        {
            double[,] rez = new double[matrix.GetUpperBound(0) + 1, matrix.GetUpperBound(1) + 1];
            for (int i = 0; i < matrix.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < matrix.GetUpperBound(1) + 1; j++)
                    rez[i, j] = (matrix[i, j] - MinIn) / (MaxIn - MinIn);
            return rez;
        }

        private void SetZeroNeurons()
        {
            for (int i = 1; i < Layers.Length; i++)
                for (int j = 0; j < Layers[i].Length; j++)
                    Layers[i][j] = 0;
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


        private double[] Sub(double[] vec1, int[] vec2 )
        {
            if (vec1.Length != vec2.Length) return null;

            double[] ans = new double[vec1.Length];
            for (int i = 0; i < vec1.Length; i++)
                ans[i] = vec1[i] - vec2[i];
            return ans;
        }
        private double[] Sub(double[] vec1, double[] vec2)
        {
            if (vec1.Length != vec2.Length) return null;

            double[] ans = new double[vec1.Length];
            for (int i = 0; i < vec1.Length; i++)
                ans[i] = vec1[i] - vec2[i];
            return ans;
        }
        private double[,] Sub(double[,] vec1, double[,] vec2)
        {
            if (vec1.GetUpperBound(0) + 1 != vec2.GetUpperBound(0) + 1 || vec1.GetUpperBound(1) + 1 != vec2.GetUpperBound(1) + 1) return null;

            double[,] ans = new double[vec1.GetUpperBound(0) + 1, vec1.GetUpperBound(1) + 1];
            for (int i = 0; i < vec1.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < vec1.GetUpperBound(1) + 1; j++)
                    ans[i, j] = vec1[i,j] - vec2[i,j];
            return ans;
        }

        private double[] Mult(double[] vec1, int value)
        {
            double[] ans = new double[vec1.Length];
            for (int i = 0; i < vec1.Length; i++)
                ans[i] = vec1[i] * value;
            return ans;
        }
        private double[] Mult(double[] vec1, double value)
        {
            double[] ans = new double[vec1.Length];
            for (int i = 0; i < vec1.Length; i++)
                ans[i] = vec1[i] * value;
            return ans;
        }
        private double[,] Mult(double[,] vec1, double value)
        {
            double[,] ans = new double[vec1.GetUpperBound(0) + 1, vec1.GetUpperBound(1) + 1];
            for (int i = 0; i < vec1.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < vec1.GetUpperBound(1) + 1; j++)
                    ans[i, j] = vec1[i, j] * value;
            return ans;
        }
        private double[] Mult(double[] vec1, double[,] vec2)
        {
            if (vec1.Length != vec2.GetUpperBound(0) + 1)
                Console.WriteLine("Mult[][,]: Несовпадение размеров");

            double[] ans = new double[vec2.GetUpperBound(1) + 1];

            for (int i = 0; i < vec2.GetUpperBound(1) + 1; i++)
                for (int j = 0; j < vec1.Length; j++)
                    ans[i] += vec1[j] * vec2[j, i];
            return ans;
        }
        private double[,] Mult(double[,] vec1, double[,] vec2)
        {
            if (vec1.GetUpperBound(1) + 1 != vec2.GetUpperBound(0) + 1)
                Console.WriteLine("Mult[,][,]: Несовпадение размеров");

            double[,] ans = new double[vec1.GetUpperBound(0) + 1, vec2.GetUpperBound(1) + 1];

            for (int i = 0; i < vec1.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < vec2.GetUpperBound(1) + 1; j++)
                    for (int k = 0; k < vec2.GetUpperBound(0) + 1; k++)
                        ans[i, j] += vec1[i, k] * vec2[k, j];
            return ans;
        }
        private double[,] Mult(double[,] vec1, double[] vec2)
        {
            if (vec1.GetUpperBound(1) + 1 != 1)
            {
                Console.WriteLine("Mult[,][]: Несовпадение размеров");
            }

            double[,] ans = new double[vec1.GetUpperBound(0) + 1, vec2.Length];

            for (int i = 0; i < vec1.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < vec2.Length; j++)
                        ans[i, j] = vec1[i, 0] * vec2[j];
            return ans;
        }
        private double[] MultEl(double[] vec1, double[] vec2)
        {
            if (vec1.Length != vec2.Length)
                Console.WriteLine("MultEl: Несовпадение размеров");

            double[] ans = new double[vec1.Length];

            for (int i = 0; i < vec1.Length; i++)
                ans[i] = vec1[i] * vec2[i];
            return ans;
        }
        private double[,] MultEl(double[,] mat1, double[,] mat2)
        {
            if (mat1.GetUpperBound(0) != mat2.GetUpperBound(0) || mat1.GetUpperBound(1) != mat2.GetUpperBound(1))
                Console.WriteLine("MultEl[,]: Несовпадение размеров");

            double[,] ans = new double[mat1.GetUpperBound(0) + 1, mat1.GetUpperBound(1) + 1];

            for (int i = 0; i < mat1.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < mat1.GetUpperBound(1) + 1; j++)
                    ans[i, j] = mat1[i, j] * mat2[i, j];
            return ans;
        }

        private double[,] Summ(double[,] matrix, double[] vector)
        {
            if ((matrix.GetUpperBound(1) + 1) != vector.Length)
                Console.WriteLine("Summ[,]: Некорректные размеры");

            double[,] rez = new double[matrix.GetUpperBound(0) + 1, matrix.GetUpperBound(1) + 1];

            for (int i = 0; i < matrix.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < matrix.GetUpperBound(1) + 1; j++)
                    rez[i, j] = vector[j] + matrix[i, j];

            return rez;
        } 

        private double[] Summ(double[,] matrix)
        {
            double[] rez = new double[matrix.GetUpperBound(1) + 1];

            for (int i = 0; i < rez.Length; i++)
                for (int j = 0; j < matrix.GetUpperBound(0) + 1; j++)
                    rez[i] += matrix[j, i];

            return rez;
        }


        private double[,] Trans(double[,] vec)
        {
            double[,] newVec = new double[vec.GetUpperBound(1) + 1, vec.GetUpperBound(0) + 1];

            for (int i = 0; i < vec.GetUpperBound(0) + 1; i++)
                for (int j = 0; j < vec.GetUpperBound(1) + 1; j++)
                    newVec[j, i] = vec[i,j];

            return newVec;
        }
        private double[,] Trans(double[] vec)
        {
            double[,] newVec = new double[vec.Length, 1];

            for (int i = 0; i < vec.Length; i++)
                    newVec[i, 0] = vec[i];

            return newVec;
        }

        private void Shuffle()
        {
            Random r = new Random();
            for (int i = buffL.Length - 1; i >= 1; i--)
            {
                int j = r.Next(i + 1);
                var temp = buffL[j];
                buffL[j] = buffL[i];
                buffL[i] = temp;
                var temp2 = buffL[j];
                buffL[j] = buffL[i];
                buffL[i] = temp2;
            }
        }


        private void SetInputLayer(double[] inputLayer)
        {
            if (inputLayer.Length == Layers[0].Length)
            {
                for (int i = 0; i < inputLayer.Length; i++)
                    Layers[0][i] = Normalize(inputLayer[i], MinIn, MaxIn);
            }
            else
                Console.WriteLine("Неверные размеры массива!");
        }

        private void SetInputLayer(byte[] inputLayer)
        {
            if (inputLayer.Length == Layers[0].Length)
            {
                for (int i = 0; i < inputLayer.Length; i++)
                    Layers[0][i] = Normalize(inputLayer[i], MinIn, MaxIn);
            }
            else
                Console.WriteLine("Неверные размеры массива!");
        }
    }
}
