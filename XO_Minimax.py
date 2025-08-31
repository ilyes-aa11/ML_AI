from random import randint
from time import sleep

# Unbeatable XO bot by utilizing the Minimax algorithm

def evalBoard(board):
    # returns 1 for player1 i.e X_player winning
    # returns -1 for player2 i.e O_player winning
    # returns 0 for tie
    # returns 2 for none of the above
    for i in range(3):
        if board[i][0] == board[i][1] and board[i][1] == board[i][2] and board[i][0] != ".":
            if board[i][0] == 'X':
                return 1
            elif board[i][0] == 'O':
                return -1
        elif board[0][i] == board[1][i] and board[1][i] == board[2][i] and board[0][i] != ".":
            if board[0][i] == 'X':
                return 1
            elif board[0][i] == 'O':
                return -1
            
    if board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] != ".":
        if board[0][0] == 'X':
            return 1
        elif board[0][0] == 'O':
            return -1
    elif board[0][2] == board[1][1] and board[1][1] == board[2][0] and board[0][2] != ".":
        if board[0][2] == 'X':
            return 1
        elif board[0][2] == 'O':
            return -1
        
    for i in range(3):
        for j in range(3):
            if board[i][j] == ".":
                return 2
            
    return 0



def minimax(player: int,board: list):
    # player == 1 => X player => maximizing player
    # player == 0 => O player => minimizing player

    state = evalBoard(board)
    if state != 2:
        return (state,None,None)
    
    if player == 1:
        optimal_move = (-1000,-1,-1)
        for i in range(3):
            for j in range(3):
                if board[i][j] == ".":
                    board[i][j] = "X"
                    res = minimax(0,board)
                    board[i][j] = "."
                    optimal_move = (res[0],i,j) if res[0] > optimal_move[0] else optimal_move
        return optimal_move

    elif player == 0:
        optimal_move = (1000,-1,-1)
        for i in range(3):
            for j in range(3):
                if board[i][j] == ".":
                    board[i][j] = "O"
                    res = minimax(1,board)
                    board[i][j] = "."
                    optimal_move = (res[0],i,j) if res[0] < optimal_move[0] else optimal_move
        return optimal_move
    


def tictactoe():

    def printBoard(board: list):
        for i in range(3):
            print(board[i][0], board[i][1], board[i][2])

    board = [["." for j in range(3)] for i in range(3)]
    player = randint(0,1)
    # bot player is 0
    winner = 2
    while (winner := evalBoard(board)) == 2:
        printBoard(board)
        if player == 1:
            while True:
                row = int(input("enter row position: "))
                col = int(input("enter column postion: "))
                if row <= 2 and row >= 0 and col <= 2 and col >= 0 and board[row][col] == ".":
                    board[row][col] = "X"
                    break
            player = 0
        else:
            print("Bot is thinking...")
            sleep(1.2)
            score , row , col = minimax(player,board)
            board[row][col] = "O"
            player = 1

    printBoard(board)
    match winner:
        case -1:
            print("Bot player won")
        case 0:
            print("Tie")
        case 1:
            print("Human player won")


tictactoe()
