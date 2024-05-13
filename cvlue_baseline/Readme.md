# Baseline 代码参考

当前文件夹内，提供 4 个任务的 baseline 代码以及原始数据样例，供大家参考。

其中，baseline 参考 CCLM(https://github.com/zengyan-97/CCLM) 与 X2VLM(https://github.com/zengyan-97/X2-VLM) 进行实现。

原始数据的样例，采样自本次比赛的训练集，最终数据集格式也与之大体相同。

### 开发集结果

<table>
    <tr>
        <th>任务</th>
        <th>指标</th>
        <th>CCLM</th>
        <th>X<sup>2</sup>VLM</th>
    </tr>
    <tr>
        <td rowspan="3">TR</td>
        <td>R@1</td>
        <td>59.8</td>
        <td>66.2</td>
    </tr>
    <tr>
        <td>R@5</td>
        <td>85.8</td>
        <td>88.5</td>
    </tr>
    <tr>
        <td>R@10</td>
        <td>91.3</td>
        <td>93.3</td>
    </tr>
    <tr>
        <td rowspan="3">IR</td>
        <td>R@1</td>
        <td>43.3</td>
        <td>48.7</td>
    </tr>
    <tr>
        <td>R@5</td>
        <td>73.7</td>
        <td>77.6</td>
    </tr>
    <tr>
        <td>R@10</td>
        <td>84.0</td>
        <td>87.5</td>
    </tr>
    <tr>
        <td>VQA</td>
        <td>Acc</td>
        <td>58.9</td>
        <td>54.6</td>
    </tr>
    <tr>
        <td>VG</td>
        <td>IoU</td>
        <td>38.6</td>
        <td>48.4</td>
    </tr>
    <tr>
        <td rowspan="3">VD</td>
        <td>R@1</td>
        <td>34.3</td>
        <td>29.3</td>
    </tr>
    <tr>
        <td>R@5</td>
        <td>49.5</td>
        <td>42.7</td>
    </tr>
    <tr>
        <td>R@10</td>
        <td>55.9</td>
        <td>49.4</td>
    </tr>
</table>


### 测试集结果

<table>
    <tr>
        <th>任务</th>
        <th>指标</th>
        <th>CCLM</th>
        <th>X2VLM</th>
    </tr>
    <tr>
        <td rowspan="3">TR</td>
        <td>R@1</td>
        <td>49.9</td>
        <td>54.8</td>
    </tr>
    <tr>
        <td>R@5</td>
        <td>75.2</td>
        <td>79.5</td>
    </tr>
    <tr>
        <td>R@10</td>
        <td>82.8</td>
        <td>86.8</td>
    </tr>
    <tr>
        <td rowspan="3">IR</td>
        <td>R@1</td>
        <td>32.0</td>
        <td>36.6</td>
    </tr>
    <tr>
        <td>R@5</td>
        <td>58.3</td>
        <td>63.4</td>
    </tr>
    <tr>
        <td>R@10</td>
        <td>69.6</td>
        <td>73.6</td>
    </tr>
    <tr>
        <td>VQA</td>
        <td>Acc</td>
        <td>58.5</td>
        <td>53.0</td>
    </tr>
    <tr>
        <td>VG</td>
        <td>IoU</td>
        <td>39.1</td>
        <td>48.8</td>
    </tr>
    <tr>
        <td rowspan="3">VD</td>
        <td>R@1</td>
        <td>32.2</td>
        <td>27.6</td>
    </tr>
    <tr>
        <td>R@5</td>
        <td>46.6</td>
        <td>41.0</td>
    </tr>
    <tr>
        <td>R@10</td>
        <td>53.3</td>
        <td>47.8</td>
    </tr>
</table>
