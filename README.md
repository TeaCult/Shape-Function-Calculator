# Shape Function Calculator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Higher-Degree Shape Functions for Tetrahedral and Triangular Elements

These Python scripts calculate higher-degree shape functions for tetrahedral and triangular finite elements which are used in FEM: 
There is ongoing research in PINNs (Physics-informed neural networks) and GPU accelerations in FEM using higher degree shape functions.
These scripts may prove useful to try some experiments on higher degree elements. 

[FEM](https://en.wikipedia.org/wiki/Finite_element_method)

[PINNS](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)

## Usage
To use the scripts, simply change the degree parameter in the code to your desired value and run it in IPython or any IDE you prefer.
Set the degree to your desired value: `degree = n  # Replace 'n' with the desired degree`

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Attribution Requirement:

You are required to provide attribution to the original author in any use of this software. Please include the following:

The original author's name: Gediz GÜRSU
A link to the original repository: Shape Function Calculator
An indication if changes were made.
Example attribution:
`This software uses code from Shape Function Calculator by Gediz GÜRSU (https://github.com/TeaCult/Shape-Function-Calculator).`

## Results for tethrahedral 4th degree Shape Functions: 

### Visual
![image](https://github.com/user-attachments/assets/41f841cc-918e-4751-bc51-9d874abff880)

### Shape Functions:
$$N1= \frac{32 z^{4}}{3} - 16 z^{3} + \frac{22 z^{2}}{3} - z$$
$$N2= \frac{128 y z^{3}}{3} - 32 y z^{2} + \frac{16 y z}{3}$$
$$N3= 64 y^{2} z^{2} - 16 y^{2} z - 16 y z^{2} + 4 y z$$
$$N4= \frac{128 y^{3} z}{3} - 32 y^{2} z + \frac{16 y z}{3}$$
$$N5= \frac{32 y^{4}}{3} - 16 y^{3} + \frac{22 y^{2}}{3} - y$$
$$N6= \frac{128 x z^{3}}{3} - 32 x z^{2} + \frac{16 x z}{3}$$
$$N7= 128 x y z^{2} - 32 x y z$$
$$N8= 128 x y^{2} z - 32 x y z$$
$$N9= \frac{128 x y^{3}}{3} - 32 x y^{2} + \frac{16 x y}{3}$$
$$N10= 64 x^{2} z^{2} - 16 x^{2} z - 16 x z^{2} + 4 x z$$
$$N11= 128 x^{2} y z - 32 x y z$$
$$N12= 64 x^{2} y^{2} - 16 x^{2} y - 16 x y^{2} + 4 x y$$
$$N13= \frac{128 x^{3} z}{3} - 32 x^{2} z + \frac{16 x z}{3}$$
$$N14= \frac{128 x^{3} y}{3} - 32 x^{2} y + \frac{16 x y}{3}$$
$$N15= \frac{32 x^{4}}{3} - 16 x^{3} + \frac{22 x^{2}}{3} - x$$
$$N16= - \frac{128 x z^{3}}{3} + 32 x z^{2} - \frac{16 x z}{3} - \frac{128 y z^{3}}{3} + 32 y z^{2} - \frac{16 y z}{3} - \frac{128 z^{4}}{3} + \frac{224 z^{3}}{3} - \frac{112 z^{2}}{3} + \frac{16 z}{3}$$
$$N17= - 128 x y z^{2} + 32 x y z - 128 y^{2} z^{2} + 32 y^{2} z - 128 y z^{3} + 160 y z^{2} - 32 y z$$
$$N18= - 128 x y^{2} z + 32 x y z - 128 y^{3} z - 128 y^{2} z^{2} + 160 y^{2} z + 32 y z^{2} - 32 y z$$
$$N19= - \frac{128 x y^{3}}{3} + 32 x y^{2} - \frac{16 x y}{3} - \frac{128 y^{4}}{3} - \frac{128 y^{3} z}{3} + \frac{224 y^{3}}{3} + 32 y^{2} z - \frac{112 y^{2}}{3} - \frac{16 y z}{3} + \frac{16 y}{3}$$
$$N20= - 128 x^{2} z^{2} + 32 x^{2} z - 128 x y z^{2} + 32 x y z - 128 x z^{3} + 160 x z^{2} - 32 x z$$
$$N21= - 256 x^{2} y z - 256 x y^{2} z - 256 x y z^{2} + 256 x y z$$
$$N22= - 128 x^{2} y^{2} + 32 x^{2} y - 128 x y^{3} - 128 x y^{2} z + 160 x y^{2} + 32 x y z - 32 x y$$
$$N23= - 128 x^{3} z - 128 x^{2} y z - 128 x^{2} z^{2} + 160 x^{2} z + 32 x y z + 32 x z^{2} - 32 x z$$
$$N24= - 128 x^{3} y - 128 x^{2} y^{2} - 128 x^{2} y z + 160 x^{2} y + 32 x y^{2} + 32 x y z - 32 x y$$
$$N25= - \frac{128 x^{4}}{3} - \frac{128 x^{3} y}{3} - \frac{128 x^{3} z}{3} + \frac{224 x^{3}}{3} + 32 x^{2} y + 32 x^{2} z - \frac{112 x^{2}}{3} - \frac{16 x y}{3} - \frac{16 x z}{3} + \frac{16 x}{3}$$
$$N26= 64 x^{2} z^{2} - 16 x^{2} z + 128 x y z^{2} - 32 x y z + 128 x z^{3} - 144 x z^{2} + 28 x z + 64 y^{2} z^{2} - 16 y^{2} z + 128 y z^{3} - 144 y z^{2} + 28 y z + 64 z^{4} - 128 z^{3} + 76 z^{2} - 12 z$$
$$N27= 128 x^{2} y z + 256 x y^{2} z + 256 x y z^{2} - 224 x y z + 128 y^{3} z + 256 y^{2} z^{2} - 224 y^{2} z + 128 y z^{3} - 224 y z^{2} + 96 y z$$
$$N28= 64 x^{2} y^{2} - 16 x^{2} y + 128 x y^{3} + 128 x y^{2} z - 144 x y^{2} - 32 x y z + 28 x y + 64 y^{4} + 128 y^{3} z - 128 y^{3} + 64 y^{2} z^{2} - 144 y^{2} z + 76 y^{2} - 16 y z^{2} + 28 y z - 12 y$$
$$N29= 128 x^{3} z + 256 x^{2} y z + 256 x^{2} z^{2} - 224 x^{2} z + 128 x y^{2} z + 256 x y z^{2} - 224 x y z + 128 x z^{3} - 224 x z^{2} + 96 x z$$
$$N30= 128 x^{3} y + 256 x^{2} y^{2} + 256 x^{2} y z - 224 x^{2} y + 128 x y^{3} + 256 x y^{2} z - 224 x y^{2} + 128 x y z^{2} - 224 x y z + 96 x y$$
$$N31= 64 x^{4} + 128 x^{3} y + 128 x^{3} z - 128 x^{3} + 64 x^{2} y^{2} + 128 x^{2} y z - 144 x^{2} y + 64 x^{2} z^{2} - 144 x^{2} z + 76 x^{2} - 16 x y^{2} - 32 x y z + 28 x y - 16 x z^{2} + 28 x z - 12 x$$
$$N32= - \frac{128 x^{3} z}{3} - 128 x^{2} y z - 128 x^{2} z^{2} + 96 x^{2} z - 128 x y^{2} z - 256 x y z^{2} + 192 x y z - 128 x z^{3} + 192 x z^{2} - \frac{208 x z}{3} - \frac{128 y^{3} z}{3} - 128 y^{2} z^{2} + 96 y^{2} z - 128 y z^{3} + 192 y z^{2} - \frac{208 y z}{3} - \frac{128 z^{4}}{3} + 96 z^{3} - \frac{208 z^{2}}{3} + 16 z$$
$$N33= - \frac{128 x^{3} y}{3} - 128 x^{2} y^{2} - 128 x^{2} y z + 96 x^{2} y - 128 x y^{3} - 256 x y^{2} z + 192 x y^{2} - 128 x y z^{2} + 192 x y z - \frac{208 x y}{3} - \frac{128 y^{4}}{3} - 128 y^{3} z + 96 y^{3} - 128 y^{2} z^{2} + 192 y^{2} z - \frac{208 y^{2}}{3} - \frac{128 y z^{3}}{3} + 96 y z^{2} - \frac{208 y z}{3} + 16 y$$
$$N34= - \frac{128 x^{4}}{3} - 128 x^{3} y - 128 x^{3} z + 96 x^{3} - 128 x^{2} y^{2} - 256 x^{2} y z + 192 x^{2} y - 128 x^{2} z^{2} + 192 x^{2} z - \frac{208 x^{2}}{3} - \frac{128 x y^{3}}{3} - 128 x y^{2} z + 96 x y^{2} - 128 x y z^{2} + 192 x y z - \frac{208 x y}{3} - \frac{128 x z^{3}}{3} + 96 x z^{2} - \frac{208 x z}{3} + 16 x$$
$$N35= \frac{32 x^{4}}{3} + \frac{128 x^{3} y}{3} + \frac{128 x^{3} z}{3} - \frac{80 x^{3}}{3} + 64 x^{2} y^{2} + 128 x^{2} y z - 80 x^{2} y + 64 x^{2} z^{2} - 80 x^{2} z + \frac{70 x^{2}}{3} + \frac{128 x y^{3}}{3} + 128 x y^{2} z - 80 x y^{2} + 128 x y z^{2} - 160 x y z + \frac{140 x y}{3} + \frac{128 x z^{3}}{3} - 80 x z^{2} + \frac{140 x z}{3} - \frac{25 x}{3} + \frac{32 y^{4}}{3} + \frac{128 y^{3} z}{3} - \frac{80 y^{3}}{3} + 64 y^{2} z^{2} - 80 y^{2} z + \frac{70 y^{2}}{3} + \frac{128 y z^{3}}{3} - 80 y z^{2} + \frac{140 y z}{3} - \frac{25 y}{3} + \frac{32 z^{4}}{3} - \frac{80 z^{3}}{3} + \frac{70 z^{2}}{3} - \frac{25 z}{3} + 1$$

### Nodes:
$$Node1=  (0, 0, 1)$$
$$Node2=  (0, 1/4, 3/4)$$
$$Node3=  (0, 1/2, 1/2)$$
$$Node4=  (0, 3/4, 1/4)$$
$$Node5=  (0, 1, 0)$$
$$Node6=  (1/4, 0, 3/4)$$
$$Node7=  (1/4, 1/4, 1/2)$$
$$Node8=  (1/4, 1/2, 1/4)$$
$$Node9=  (1/4, 3/4, 0)$$
$$Node10=  (1/2, 0, 1/2)$$
$$Node11=  (1/2, 1/4, 1/4)$$
$$Node12=  (1/2, 1/2, 0)$$
$$Node13=  (3/4, 0, 1/4)$$
$$Node14=  (3/4, 1/4, 0)$$
$$Node15=  (1, 0, 0)$$
$$Node16=  (0, 0, 3/4)$$
$$Node17=  (0, 1/4, 1/2)$$
$$Node18=  (0, 1/2, 1/4)$$
$$Node19=  (0, 3/4, 0)$$
$$Node20=  (1/4, 0, 1/2)$$
$$Node21=  (1/4, 1/4, 1/4)$$
$$Node22=  (1/4, 1/2, 0)$$
$$Node23=  (1/2, 0, 1/4)$$
$$Node24=  (1/2, 1/4, 0)$$
$$Node25=  (3/4, 0, 0)$$
$$Node26=  (0, 0, 1/2)$$
$$Node27=  (0, 1/4, 1/4)$$
$$Node28=  (0, 1/2, 0)$$
$$Node29=  (1/4, 0, 1/4)$$
$$Node30=  (1/4, 1/4, 0)$$
$$Node31=  (1/2, 0, 0)$$
$$Node32=  (0, 0, 1/4)$$
$$Node33=  (0, 1/4, 0)$$
$$Node34=  (1/4, 0, 0)$$
$$Node35=  (0, 0, 0)$$

# Dörtyüzlü ve Üçgen Elemanlar için Yüksek Dereceli Şekil Fonksiyonları
Bu Python script'leri, FEM'de (Sonlu Elemanlar Yöntemi) kullanılan dörtyüzlü ve üçgen sonlu elemanlar için yüksek dereceli şekil fonksiyonlarını hesaplar.
PINN'ler (Fizik Bilgili Sinir Ağları) ve FEM'de yüksek dereceli şekil fonksiyonlarını kullanan GPU hızlandırmaları üzerine devam eden araştırmalar vardır. 
Bu script'ler, yüksek dereceli elemanlar üzerinde bazı deneyler yapmak için faydalı olabilir.

[FEM](https://en.wikipedia.org/wiki/Finite_element_method)

[PINNS](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)

## Kullanım
Betikleri kullanmak için, kodda bulunan degree parametresini istediğiniz değere değiştirin ve IPython veya tercih ettiğiniz herhangi bir IDE'de çalıştırın.
İstediğiniz dereceyi ayarlayın:  `degree = n # 'n' yerine istediğiniz dereceyi yazın`

## Lisans
Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için `LICENSE` dosyasına bakın.

Atıf Gereksinimi:

Bu yazılımın herhangi bir kullanımında orijinal yazara atıf yapmanız gerekmektedir. Lütfen aşağıdakileri ekleyin:

Orijinal yazarın adı: Gediz GÜRSU
Orijinal depo bağlantısı: Shape Function Calculator
Değişiklik yapıldıysa belirtiniz.

Örnek Atıf
`Bu yazılım, Gediz GÜRSU tarafından geliştirilen Shape Function Calculator'dan alınmıştır (https://github.com/TeaCult/Shape-Function-Calculator).`

