import usb_cdc
import usb_hid
import usb_midi
import storage

try:
    usb_hid.disable()
    usb_midi.disable()
except:
    pass

try:
    usb_cdc.enable(console=True, data=True)
except:
    pass

storage.enable()