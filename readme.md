# <p align="center"><font size=50><strong>CVLUE: Chinese Vision-Language Understanding Evaluation</strong></font></p>

[Chinese Version](readme_zh.md)

## Task Introduction

The Chinese Vision-Language Understanding Evaluation (CVLUE) task aims to evaluate the vision-language modeling and understanding capabilities of Chinese vision-language pre-training models from multiple perspectives, including Image-Text Retrieval, Visual Question Answering, Visual Grounding, and Visual Dialog. This task includes the following five sub-tasks:

- **Image Retrieval**: Retrieve the corresponding image from several candidates based on the given text description.
- **Text Retrieval**: Retrieve the corresponding text description from several candidates based on the given image.
- **Visual Question Answering**: Answer questions based on the given image using short phrases.
- **Visual Grounding**: Identify the corresponding entity in the image based on the given image and text description.
- **Visual Dialog**: Select the most appropriate reply text from several candidates based on the given image and dialog history.

## Evaluation Data

This task includes images from the following 15 major categories and 92 subcategories. The images are manually collected according to the following categories, with strict requirements that the content of the images is representative of or commonly seen in the Chinese cultural environment:

<center>
    <table>
        <tr>
        <td align="center"> <b>Major Category</b></td>
        <td align="center"> <b>Subcategories</b></td>
        <td align="center"> <b>Number of Subcategories</b></td> 
        </tr>
        <tr>
        <td align="center">Animals</td>
        <td align="center">Panda, Cow, Fish, Dog, Horse, Chicken, Mouse, Bird, Human, Cat</td>
        <td align="center">10</td>
        </tr>
        <tr>
        <td align="center">Food</td>
        <td align="center">Hot pot, Rice, Dumplings, Noodles, Buns</td>
        <td align="center">5</td>
        </tr>
        <tr>
        <td align="center">Drinks</td>
        <td align="center">Bubble tea, Cola, Milk, Tea, Porridge, Alcohol</td>
        <td align="center">6</td>
        </tr>
        <tr>
        <td align="center">Clothes</td>
        <td align="center">Hanfu, Tang suit, Cheongsam, Suit, T-shirt</td>
        <td align="center">5</td>
        </tr>
        <tr>
        <td align="center">Plants</td>
        <td align="center">Willow, Ginkgo, Chinese parasol, Birch, Pine, Chrysanthemum, Peony, Orchid, Lotus, Lily</td>
        <td align="center">10</td>
        </tr>
        <tr>
        <td align="center">Fruits</td>
        <td align="center">Lychee, Hawthorn, Apple, Cantaloupe, Longan</td>
        <td align="center">5</td>
        </tr>
        <tr>
        <td align="center">Vegetables</td>
        <td align="center">Bok choy, Potato, Chinese Cabbage, Carrot, Cauliflower</td>
        <td align="center">5</td>
        </tr>
        <tr>
        <td align="center">Agriculture</td>
        <td align="center">Hoe, Plow, Harrow, Sickle, Carrying pole</td>
        <td align="center">5</td>
        </tr>
        <tr>
        <td align="center">Tools</td>
        <td align="center">Spoon, Bowl, Cutting board, Chopsticks, Wok, Fan, Chinese Cleaver, Wok Spatula</td>
        <td align="center">8</td>
        </tr>
        <tr>
        <td align="center">Furniture</td>
        <td align="center">TV, Table, Chair, Refrigerator, Stove</td>
        <td align="center">5</td>
        </tr>
        <tr>
        <td align="center">Sports</td>
        <td align="center">Ping-Pong, Basketball, Swimming, Football, Running</td>
        <td align="center">5</td>
        </tr>
        <tr>
        <td align="center">Celebrations</td>
        <td align="center">Lion Dance, Dragon Boat, National Flag, Mooncake, Couplets, Lantern</td>
        <td align="center">6</td>
        </tr>
        <tr>
        <td align="center">Education</td>
        <td align="center">Pencil, Blackboard, Chinese Brush, Chalk, Ballpoint, Scissors</td>
        <td align="center">6</td>
        </tr>
        <tr>
        <td align="center">Musical Instruments</td>
        <td align="center">Guzheng, Erhu, Suona, Drum, Pipa</td>
        <td align="center">5</td>
        </tr>
        <tr>
        <td align="center">Art</td>
        <td align="center">Calligraphy, Chinese Shadow Play, Paper Cutting, Terracotta Army, Ding, Ceramics</td>
        <td align="center">6</td>
        </tr>
    </table> 
</center>


## Data Examples

Next, data examples for each task category are provided.

### Image-Text Retrieval

Each image has 5 different descriptions.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="example/图文检索.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Image-Text Retrieval</div>
</center>

The 5 captions are:

1. There is a hot pot placed in the middle of the table.
2. A two-flavor hot pot is placed on a wooden table.
3. A hot pot with one spicy and one mushroom broth base is placed on the table.
4. The hot pot is surrounded by various ingredients for hot pot, such as vegetables, meat, and meatballs.
5. In the middle of the table, there is a two-flavor hot pot, and the ceramic bowls around it contain ingredients for hot pot.

### Visual Question Answering

Ask questions based on the image and provide answers.


<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="example/视觉问答.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Visual Question Answering</div>
</center>

The 3 Questions and Answers are：
- Q: In which direction is the dragon boat heading?<br>A: To the right
- Q: How many teams are rowing dragon boats?<br>A: 5
- Q: Are most people standing or sitting?<br>A: Sitting

### Visual Grounding

Provide descriptions of certain entities in the image and their corresponding bounding boxes (marked with rectangles in the image).

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="example/视觉定位.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Visual Grounding</div>
</center>

Description of entities:
1. The shadow puppet held by the girl wearing glasses
2. The shadow puppet held by the boy with short hair

### Visual Dialog

Provide an image and its description, then conduct a question-answer dialog based on the image.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="example/视觉对话.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Visual Dialog</div>
</center>

- Caption: There is a lot of food on the blue table mat<br>
- Q1: What foods are on the table?<br>A1: The foods include eggs, buns, side dishes, steamed buns, and porridge
- Q2: What kind of porridge is on the table?<br>A2: The porridge on the table is black rice porridge<br>
......
- Q10: How many eggs are on the table?<br>A10: There are two eggs on the table

## Evaluation Metrics

The evaluation metrics for each sub-task are as follows:

### Image-Text Retrieval

For the image-text retrieval task, the evaluation metric is Recall $ R@k（k = 1, 5, 10）$.

$$ R@k=\frac{Number\ of\ correct\ results\ in\ the\ top\ k\ retrieval\ rankings}{Total\ number\ of\ samples} $$

### Visual Question Answering

For the visual question answering task, the evaluation metric is straightforward, namely the accuracy of answering questions $ Accuracy $.

$$ Accuracy=\frac{Number\ of\ correct\ answers}{Total\ number\ of\ questions} $$

### Visual Grounding

For the visual grounding task, the evaluation metrics are based on Intersection over Union $ IoU $. The metrics include the alignment accuracy of images (an $ IoU $ value greater than 0.5 is considered correct) and the mean $ IoU $.

$$ IoU=\frac{Area\ of\ overlap\ between\ predicted\ region\ and\ ground\ truth\ region}{Area\ of\ union\ of\ predicted\ region\ and\ ground\ truth\ region} $$
$$ IoU_{Accuracy}=\frac{Number\ of\ samples\ with\ IoU\ over\ 0.5}{Total\ number\ of\ grounding\ samples} $$
$$ \overline{IoU}=\frac{Sum\ of\ IoU\ of\ all\ predictions}{Total\ number\ of\ grounding\ samples} $$

### Visual Dialog

For the visual dialog task, the evaluation metric is Recall $ R@k（k = 1, 5, 10）$.

$$ R@k=\frac{Number\ of\ correct\ results\ in\ the\ top\ k\ retrieval\ rankings}{Total\ number\ of\ samples} $$


## Data Access and Citation

The CVLUE data is freely available upon request under the CC BY-NC-ND 4.0 license. Please submit your request via [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfPfYq0bhjG4QRjssefrD5jM4F8obhYQG1XQxlaPlXqspFcAA/viewform?usp=sf_link).

If you use the CVLUE data, please cite the following paper:

```
@misc{wang-etal-2024-cvlue,
    title={CVLUE: A New Benchmark Dataset for Chinese Vision-Language Understanding Evaluation},
    author={Yuxuan Wang and Yijun Liu and Fei Yu and Chen Huang and Kexin Li and Zhiguo Wan and Wanxiang Che and Hongyang Chen},
    year={2024},
    eprint={2407.01081},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
