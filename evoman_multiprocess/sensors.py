import numpy
import struct
import binascii

# sensors for the controllers
class Sensors():


    def get(self, game, player, enemy):


        # calculates vertical and horizontal distances between sprites centers

        posx_p = player.rect.left +((player.rect.right - player.rect.left)/2)
        posy_p = player.rect.bottom +((player.rect.top - player.rect.bottom)/2)
        posx_e = enemy.rect.left +((enemy.rect.right - enemy.rect.left)/2)
        posy_e = enemy.rect.bottom +((enemy.rect.top - enemy.rect.bottom)/2)

        # pre-allocate values for the bullets
        param_values = [ posx_p-posx_e, posy_p-posy_e, player.direction, enemy.direction] + [0]*16

        # calculates vertical and horizontal distances between player and the center of enemy's bullets
        bullet_count = 0
        for i in range(0,len(enemy.twists)):
            if enemy.twists[i] != None:
                posx_be = enemy.twists[i].rect.left +((enemy.twists[i].rect.right - enemy.twists[i].rect.left)/2)
                posy_be = enemy.twists[i].rect.bottom +((enemy.twists[i].rect.top - enemy.twists[i].rect.bottom)/2)
                param_values[4 + bullet_count * 2] = posx_p - posx_be
                param_values[4 + bullet_count * 2 + 1] = posy_p - posy_be
                bullet_count+=1


        # applies several transformations to input variables (sensors)
        if game.inputscoded == "yes":

            types = struct.Struct('q q q q q q q q q q q q q q q q q q q q') # defines the data types of each item of the array that will be packed. (q=int, f=flo)
            packed_data = types.pack(*param_values)  # packs data as struct
            coded_variables =  binascii.hexlify(packed_data)  # converts packed data to an hexadecimal string
            coded_variables = [coded_variables[i:i+2] for i in range(0, len(coded_variables), 2)] # breaks hexadecimal string in bytes.
            coded_variables = numpy.array(map(lambda y: int(y, 16), coded_variables))  # converts bytes to integer

            param_values = coded_variables


        self.sensors = param_values # defines sensors state


        return numpy.array(self.sensors)
