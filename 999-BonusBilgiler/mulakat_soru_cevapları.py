#################################################################################
#                                 Q & A                                         #
#################################################################################

# Soru 1: Interpreter oluşturma amacımız nedir?
# Python yorumlanan bir programlama dili midir? Neden?
# Cevap 1:
#
# Kaynak kodunun bir yorumlayıcı (interpreter) kullanılarak yazılmış kodun doğrudan çalıştırılması işlemidir.
# Yorumlayıcı, kaynak kodunun her bir satırını okuyarak, yorumlayarak ve hemen çalıştırarak sonucu üretir.
# Yorumlayıcı programları, kaynak kodunu doğrudan yürüten programlardır.
# Yorumlanan diller arasında Python, Ruby, PHP, Perl ve JavaScript gibi popüler diller bulunur.
# Bu dillerin avantajı, daha kolay öğrenilebilir olması ve daha hızlı kod geliştirme sağlamalarıdır.
# Ayrıca, bu diller genellikle daha esnek ve daha yüksek seviyeli programlama yapısına sahiptirler,
# bu da kodun daha okunaklı ve anlaşılır olmasını sağlar.



# # Derlenmiş programlama dilleri, kodun derlenme (bilgisayarın anlayabileceği makine kodu haline dönüştürülmesi)
# # işlemi sonucu makine (bilgisayar) diline çevrildiği programlama dilleridir.
# # Derleyici programları, kaynak kodunun derlenmesi sonucu elde edilen makine kodunu oluştururlar.Derlenmiş diller
# # arasında C, C++, Ada ve Pascal gibi popüler diller bulunur.
# # Bu dillerin kaynak kodları, derlenme işlemi sonrasında doğrudan bilgisayar tarafından anlaşılabilir
# # makine koduna dönüştürüldüğünden, genellikle daha hızlı ve daha yüksek performanslı uygulamalar oluşturmak için tercih edilirler.


##################################################################################
# Soru 2: Python neden kullanmaktayız? Avantajları ve dezavantajları nelerdir?
# Cevap 2:
#
# Avantaj
# 1. Okuması, Öğrenmesi ve Yazması Kolay
# Python çok üretken bir dildir . Python'un basitliği nedeniyle geliştiriciler sorunu çözmeye odaklanabilir.
# Python, yorumlanmış bir dildir; bu, Python'un kodu satır satır doğrudan yürüttüğü anlamına gelir .
# Herhangi bir hata durumunda, daha fazla çalışmayı durdurur ve oluşan hatayı geri bildirir.
# Python, OSI onaylı açık kaynak lisansı altında gelir . Bu yapar özgür için kullanmak ve dağıtmak .
# Kaynak kodunu indirebilir, değiştirebilir ve hatta Python sürümünüzü dağıtabilirsiniz.
# Taşınabilirlik: C/C++ gibi birçok dilde , programı farklı platformlarda çalıştırmak için kodunuzu değiştirmeniz gerekir .
# Python'da durum aynı değil. Yalnızca bir kez yazıp her yerde çalıştırırsınız.


# Dezavantaj
# Python'un yorumlanmış bir dil ve dinamik olarak yazılan bir dil olduğunu yukarıda tartıştık .
# Kodun satır satır yürütülmesi genellikle yavaş yürütmeye neden olur .

# Bellek Verimli Değil
# Geliştiriciye basitlik sağlamak için Python'un biraz ödün vermesi gerekiyor. Python programlama dili büyük miktarda bellek kullanır .

# Mobil Bilişimde Zayıf
# Python genellikle sunucu tarafı programlamada kullanılır . Aşağıdaki nedenlerden dolayı istemci tarafında veya
# mobil uygulamalarda Python'u göremiyoruz. Python bellek açısından verimli değildir ve
# diğer dillere kıyasla yavaş işlem gücüne sahiptir

# Python'un dinamik doğası, Python'un yavaş hızından da sorumludur, çünkü kodu yürütürken fazladan iş yapması gerekir.
# Bu nedenle Python, hızın projenin önemli bir yönü olduğu amaçlar için kullanılmaz.

##################################################################################
# Soru 3: Nesne yönelimli programlama(OOP) Temel Mantığı nedir?
# Cevap 3:

# Nesne yönelimli programlama (Object-Oriented Programming veya kısaca OOP), sınıflar (classes) ve
# bu sınıflardan oluşturulan nesneler (objects) temelinde çalışan bir programlama paradigmasıdır.
# OOP, karmaşıklığı azaltmak, kodu daha organize etmek ve kodun daha yeniden kullanılabilir olmasını sağlamak için kullanılır.

# Sınıflar (Classes): Sınıflar, nesnelerin (objects) tasarlandığı şablonları temsil eder.
# Sınıflar, bir nesnenin sahip olacağı özellikleri (veri alanları veya değişkenleri) ve davranışları (metodları) tanımlar.
# Örneğin, bir "Araba" sınıfı, araçların sahip olabileceği özellikleri (marka, model, hız vb.) ve metodları (sür, dur, hız artır vb.) tanımlayabilir.

# Nesneler (Objects): Sınıflardan türetilen nesneler, gerçek dünyadaki varlıkların temsilidir.
# Her nesne, kendi özelliklerine sahip olabilir ve sınıf tarafından tanımlanan metodları çağırabilir.


# Araba sınıfını tanımla
class Araba:
    def __init__(self, marka, model, hiz=0):
        self.marka = marka
        self.model = model
        self.hiz = hiz

    def sur(self, hiz):
        self.hiz = hiz
        print(f"{self.marka} {self.model}, {self.hiz} km/saat hızında sürülüyor.")

    def dur(self):
        self.hiz = 0
        print(f"{self.marka} {self.model}, durdu.")

    def hiz_artir(self, artis_miktari):
        self.hiz += artis_miktari
        print(f"{self.marka} {self.model}, hızı {artis_miktari} km/saat artırıldı. Yeni hız: {self.hiz} km/saat.")


# Araba nesnelerini oluştur ve kullan
araba1 = Araba("Toyota", "Camry")
araba2 = Araba("Honda", "Civic", 60)

araba1.sur(40)


##################################################################################
# Soru 4: _init__ neyi ifade etmektedir? Kullanmak şart mıdır?
# Cevap 4:

# __init__ , OOP ile programlamada bir class'ın yapıcı metodudur.
# Eğer bir class'tan nesne üreteceksek __init__, class'ın ilk metodu olarak tanımlanabilir. Fakat ilk sıraya tanımlama
# zorunluluğu yoktur.
# class içindee türetilen nesnelere ait özellikler bu metot ile nesneler atanır.
# self, init metodu ile gelen ve class içinde türetmiş olduğunuz nesnelere ulaşmanızı sağlayan bir kavramdır.


##################################################################################
# Soru 5: Inheritance(Miras) nedir?
# Cevap 5:

# Inheritance (Miras): Miras, bir sınıfın diğer bir sınıftan özelliklerini ve davranışlarını devralmasına olanak tanır.
# Bu, kodun yeniden kullanılabilirliğini artırır ve sınıflar arasında hiyerarşi oluşturabilir.

# Ornek:

# Ana sınıf: Hayvan
class Hayvan:
    def __init__(self, isim, yas):
        self.isim = isim
        self.yas = yas

    def ses_cikar(self):
        pass

# Alt sınıf 1: Köpek, Hayvan sınıfından miras alır
class Kopek(Hayvan):
    def ses_cikar(self):
        return "Hav! Hav!"

# # Köpek sınıflarını kullanma
dog = Kopek("Buddy", 3)
print(f"{dog.isim} ({dog.yas} yaşında) ses çıkarıyor: {dog.ses_cikar()}")


##################################################################################
# Soru 6: help() ve dir() fonksiyonları ne işe yarar?
# Cevap 6:

# Her iki fonksiyon da Python yorumlayıcısı tarafından erişilebilir.
# Dahili fonksiyonlardaki birleştirilmiş dump’ları görüntülemek için kullanılır.
#
# help() size belgeleme string’ini gösterirken dir() tanımlanmış sembolleri görüntüler.



##################################################################################
# Soru 7: "Tuple" ve "List" performans ve kullanım açısından aralarında nasıl bir farklılık vardır?
# Cevap 7:

# Listeler (Lists): Listeler, öğelerin değiştirilebilir olmasından dolayı bellek kullanımı ve
# işlemci gücü açısından tuple'lardan daha fazla kaynak tüketebilir.
# Tuplelar (Tuples): Tuple'lar değiştirilemez olduğu için daha hızlıdır ve daha az bellek tüketir.

# Listeler (Lists): Listeler, genellikle verileri dinamik olarak değiştirmek veya sıralamak için kullanılır.
# Özellikle aynı türden verileri saklamak için uygundur.
#
# Tuplelar (Tuples): Tuple'lar, verilerin sabit ve değiştirilemez olmasını istediğiniz durumlarda kullanılır.
# Özellikle bir işlem sonucunun veya işlevden birden fazla değer döndürmeniz gerektiğinde tuple'lar kullanışlıdır.


##################################################################################
# Soru 8: "Tuple" ve "List" performans ve kullanım açısından aralarında nasıl bir farklılık vardır?
# Cevap 8:

# Paket, bir yazılımın derlenmiş hali ya da kaynak kodu olabilir. Bu nedenle modül ifadesi de bu tür uygulamaları nitelendirirken
# kullanılabilir. Paket ifadesinin açıklamasında, paketin nihai bir uygulama olabileceğinden de bahsetmiştim.
# Bağımlılık (dependency), bu nihai paket öncesinde, geliştirme sürecinde kullanılan paketleri nitelendirmektedir.
# Bir uygulamanın development ve/veya production aşamalarında çalışabilmesi için gerekli olan paketlerdir.
#
# Paket Yönetim Sistemi
# Uygulama geliştirirken ya da nihai bir uygulamayı işletim sistemimize indirirken kurulum, yapılandırma ve
# uygulamanın kaldırılması gibi süreçler söz konusu olabilmektedir. Diğer yandan, uygulamaların yeni sürümlerinin
# ve olası sürüm farklılıklarının ve güncelleme notlarının takip edilmesi de ayrı bir önem arz eder.
# Bu gibi süreçlerde kararlı bir şekilde çalışan araçlara ihtiyaç duyulmaktadır. Paket (package) ve bağımlılık (dependency)
# yöneticiler (package manager ya da package-management system) tam olarak bu süreci üstlenirler.
# Ortaya çıkan gereksinimlere göre yöneticiler farklı ele alış biçimlerine sahip olabilirler.