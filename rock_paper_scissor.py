import random
import pdb

# pdb to set_trace for debugging
# locals() to get all local variables, global()
# q- quit, c - contine, h -  help

def get_choices():
    player_choice=input("Enter a choice (rock, paper, scissor)")
    if player_choice not in  ['rock','paper','scissor']:
        print('invalid choice provided')
        return 
    computer_choice=random.choice(['rock','paper','scissor'])
    #pdb.set_trace()
    choices={"player":player_choice,"computer":computer_choice}
    return choices

def select_winner(player,computer):
    print(f"You chose {player}, computer chose {computer}")
    if computer==player:
        print("It's a tie")
    elif player =='rock':
        if computer=='paper':
            print('You lose')
        else:
            print('You win')
    elif player =='paper':
        if computer=='scissor':
            print('You lose')
        else:
            print('You win')
    elif player =='scissor':
        if computer=='rock':
            print('You lose')
        else:
            print('You win')



choices=get_choices()
if choices:
    select_winner(choices['player'],choices['computer'])