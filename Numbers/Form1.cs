using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

namespace Numbers
{
    public partial class Form1 : Form
    {
        private MouseButtons mouseKey;

        private Graphics g;
        private Graphics gg;
        private SolidBrush brush = new SolidBrush(Color.White);

        private Bitmap picBox = new Bitmap(28 * 16, 28 * 16);
        private Bitmap miniPicBox = new Bitmap(28, 28);

        private int[] bigSize = new int[] { 28 * 16, 28 * 16 };
        private int[] smallSize = new int[] { 28, 28 };

        private AIv2 ai = new AIv2("relu", "sqr", new int[] { 28 * 28, 512, 256, 128, 10 });
        private ProgressBar[] progrBar;
        private Label[] progrLab;

        public Form1()
        {
            InitializeComponent();
            progrBar = new ProgressBar[] { PB0, PB1, PB2, PB3, PB4, PB5, PB6, PB7, PB8, PB9 };
            progrLab = new Label[]       { LB0, LB1, LB2, LB3, LB4, LB5, LB6, LB7, LB8, LB9 };

            g = Graphics.FromImage(picBox);
            gg = Graphics.FromImage(miniPicBox);

            g.Clear(Color.Black);
            gg.Clear(Color.Black);

            pictureBox1.Image = miniPicBox;
            pictureBox2.Image = picBox;

        }

        private void pictureBox2_MouseDown(object sender, MouseEventArgs e)
        {
            mouseKey = e.Button;

            if (mouseKey == MouseButtons.Middle)
            {
                g.Clear(Color.Black);
                gg.Clear(Color.Black);

                pictureBox2.Image = picBox;
                pictureBox1.Image = miniPicBox;

                for (int i = 0; i < 10; i++)
                    progrBar[i].Value = 0;
            }
        }

        private void pictureBox2_MouseUp(object sender, MouseEventArgs e)
        {
            mouseKey = MouseButtons.None;
        }

        private void pictureBox2_MouseMove(object sender, MouseEventArgs e)
        {
            double[] pred;

            if (mouseKey != MouseButtons.None && mouseKey != MouseButtons.Middle)
            {
                Draw(e.X, e.Y, mouseKey);
                pictureBox2.Image = picBox;
                pictureBox1.Image = miniPicBox;

                pred = ai.Predict(BmpToArr(miniPicBox));

                for (int i = 0; i < 10; i++)
                {
                    progrBar[i].Value = (int)(pred[i] * 100);
                    progrLab[i].Text = (pred[i]*100).ToString();
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            ai.Training();
        }

        private void Draw(int x, int y, MouseButtons mouseKey) 
        {
            int deltaCenter = 256 / 16;
            int deltaEdge   = deltaCenter / 2;

            if (mouseKey == MouseButtons.Right)
            {
                deltaCenter *= -1;
                deltaEdge   *= -1;
            }

            for (int i = -1; i < 2; i++)
            {
                for (int j = -1; j < 2; j++)
                {
                    int X = ((x / 16) + j) * 16;
                    int Y = ((y / 16) + i) * 16;

                    if (X >= bigSize[0] || Y >= bigSize[1] || X < 0 || Y < 0)
                        continue;

                    if (i == 0 || j == 0)
                        brush.Color = Color.FromArgb(Normalize(picBox.GetPixel(X, Y).R + deltaCenter),
                                                     Normalize(picBox.GetPixel(X, Y).G + deltaCenter),
                                                     Normalize(picBox.GetPixel(X, Y).B + deltaCenter));
                    else
                        brush.Color = Color.FromArgb(Normalize(picBox.GetPixel(X, Y).R + deltaEdge),
                                                     Normalize(picBox.GetPixel(X, Y).G + deltaEdge),
                                                     Normalize(picBox.GetPixel(X, Y).B + deltaEdge));

                    g.FillRectangle(brush, X, Y, 16, 16);
                    gg.FillRectangle(brush, X / 16, Y / 16, 1, 1);
                }
            }
        }

        private int Normalize(int num)
        {
            return Math.Max(Math.Min(num, 255), 0);
        }

        private double[] BmpToArr(Bitmap img)
        {
            double[] arr = new double[img.Height * img.Height];

            for (int y = 0; y < img.Height; y++)
                for (int x = 0; x < img.Width; x++)
                    arr[y * img.Width + x] = img.GetPixel(x, y).R;

            return arr;
        }

        private void SetPicInMnist(byte[] img)
        {
            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++)
                    miniPicBox.SetPixel(x, y, Color.FromArgb(img[y * 28 + x], img[y * 28 + x], img[y * 28 + x]));
        }
    }
}
