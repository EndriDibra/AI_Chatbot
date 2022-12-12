# Author: Endri Dibra

# importing the required libraries
import time
import datetime
import webbrowser as wb
import pyttsx3
import speech_recognition as sr


# AI voice assistant
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


# speaking ability for our AI assistant
def speak(audio):

    engine.say(audio)
    engine.runAndWait()


# this functions is to greet for
# the first time our user
def greet():

    # getting the current time
    current_time = float(datetime.datetime.now().hour)

    # our AI assistant will greet based on the current time
    if current_time >= 0.0 and current_time < 12.0:

        speak('Good morning sir')

    elif current_time > 12.0 and current_time < 18.0:

        speak('Good afternoon sir')

    else:

        speak('Good evening sir')

    speak('Nice too meet you. I am Jarvis, your Artificial Intelligence assistant.')
    speak('Together we will do some amazing things and have a great time!')
    speak('So, shall i ask you your first question ?')


# this function terminates the program
def exit():

    speak("Alright, please call me when you need me, wish you a nice day !")
    quit()


# taking commands (audio input) from user
def takecommand():

    recognize = sr.Recognizer()

    with sr.Microphone() as source:

        print('!! I am listening !!')

        recognize.pause_threshold = 0.7
        audio = recognize.listen(source)

    # error handling
    try:

        print('!! I am recognizing !!')

        # using google services for voice recognition
        command = recognize.recognize_google(audio, language='en-uk')
        print(f'User said: {command}\n')

    # in case that our AI assistant did not
    # understand our command (audio input)
    except Exception as e:

        speak('Could you say that again please')

        return 'None'

    return command


# execution control
if __name__ == '__main__':

    greet()


# executing speaker's commands
    while True:

        # converting command (text) in lower case
        command = takecommand().lower()

        if 'how are you' in command:

            speak("I am fine thanks!, enjoy your time")

            print("What's the next question ?")
            speak("What's the next question ?")


        elif 'are you there' in command:

            speak("Yes! At your service !")

            print("What's the next question ?")
            speak("What's the next question ?")


        elif 'time' in command:

            # importing the required libraries
            from datetime import datetime

            # getting the current time
            time_now = datetime.now()
            current_time = time_now.strftime("%H:%M:%S")

            print("Time: ", current_time)
            speak(f'The time right now is {current_time}')

            print("What's the next question ?")
            speak("What's the next question ?")


        elif 'social media' in command:

            speak("Please tell me which platform to open, only the name of it")
            platform = takecommand().lower()

            # in case that the value is not correct , will be re-requested
            while platform == "none":

                platform = takecommand().lower()

            # otherwise will open the website
            else:

                wb.open(f"https://www.{platform}.com/")

                # waiting for 2.5 seconds
                time.sleep(2.5)

                print("What's the next question ?")
                speak("What's the next question ?")


        elif 'weather' in command:

            speak("Which city's weather should i look for ?")
            city_name = takecommand().lower()

            # in case that the value is not correct , will be re-requested
            while city_name == 'none':

                city_name = takecommand().lower()

            # otherwise will open the website
            else:

                speak(f"Opening weather for {city_name}")
                wb.open(f'https://www.google.com/search?q={city_name}+temperature&as_qdr=all&sxsrf=ALiCzsZtvVobJTpKrWFGnsmE04k9mhpo7g%3A1661279009148&source=hp&ei=IRsFY_CeBcC_xc8P1u6t4Ao&iflsig=AJiK0e8AAAAAYwUpMUZ8SsbJpL7Yv1Jp-XRwzM7KX5N8&oq=athens+te&gs_lcp=Cgdnd3Mtd2l6EAMYADIFCAAQywEyCwguEIAEEMcBEK8BMgUIABDLATIFCAAQgAQyBQgAEMsBMgsILhDHARCvARDLATILCC4QxwEQrwEQywEyBQgAEMsBMgUIABDLATILCC4QgAQQxwEQrwE6BAguECc6BAgjECc6EQguEIAEELEDEIMBEMcBENEDOgsILhCABBCxAxCDAToLCAAQgAQQsQMQgwE6DgguEIAEELEDEMcBENEDOgoIABCxAxCDARBDOgQIABBDOgcILhDUAhBDOgQILhBDOg0ILhCxAxCDARDUAhBDOhQILhCABBCxAxCDARDHARDRAxDUAlAAWO4aYOwnaABwAHgAgAHQAYgBwguSAQUwLjguMZgBAKABAQ&sclient=gws-wiz')

                # waiting for 2.5 seconds
                time.sleep(2.5)

                print("What's the next question ?")
                speak("What's the next question ?")


        elif 'exit' in command:

            # terminating the program
            exit()