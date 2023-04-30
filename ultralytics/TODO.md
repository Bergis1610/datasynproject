## Ser norge har mye flere av D00, kanskje fjern bilder fra andre datasets med klassene D20 / D40 for å nerme norge

## Cropping 

# så på en del bilder, fant ut at mye av høyre del på bildet er noe annet enn vei (Skog, mark, osv) fremdeles var det noen av de mindre bildene som inneholdt vei, og sprekker, så vi kom frem til at det å croppe 1/3 av bildet fra høyre vil øke treningshastigheten og utelukke unødvendig data. 


## anbefalt av ultralytics 
# 300 epoker 
# høyere rez om man har mye små objekter (det har me :)), husk å kjør detect.py med samme rez som man trente på 
# så stor batch size som mulig 

token:
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