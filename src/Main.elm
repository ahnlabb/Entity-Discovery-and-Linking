module Main exposing (..)

import Browser
import Http
import Url.Builder as Url
import Dict
import Element exposing (Element, el, text, row, alignRight, fill, width, height, rgb255, spacing, centerY, padding, none, px)
import Element.Input as Input
import Element.Background as Background
import Element.Border as Border
import Element.Font as Font
import Json.Decode as Decode exposing (Decoder, int, string, dict, list)
import Json.Decode.Pipeline exposing (required)
import Json.Encode as Encode
import Time


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        }



-- MODEL


type alias Model =
    { text : String
    , result : String
    , change : Change
    }


type alias Docforia =
    { text : String
    , properties : Properties
    , edges : List Edges
    , nodes : List Nodes
    }


type alias Edges =
    { variants : List String
    , layer : String
    , properties : List Properties
    , connections : List Int
    }


type alias Nodes =
    { variants : List String
    , layer : String
    , properties : List Properties
    , ranges : List Int
    }


type alias Properties =
    Dict.Dict String String


type Change
    = Waiting
    | Changed
    | Unchanged


init : () -> ( Model, Cmd Msg )
init _ =
    ( Model "" "" Waiting, Cmd.none )



-- UPDATE


type Msg
    = NewDocforia (Result Http.Error Docforia)
    | EditedText String
    | NewEl (Result Http.Error String)
    | Tick Time.Posix


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NewDocforia data ->
            ( model, Cmd.none )

        EditedText newText ->
            ( { model | text = newText, change = Changed }, Cmd.none )

        NewEl result ->
            case result of
                Ok newResult ->
                    ( { model | result = newResult }, Cmd.none )

                Err _ ->
                    ( model, Cmd.none )

        Tick _ ->
            case model.change of
                Waiting ->
                    ( model, Cmd.none )

                Changed ->
                    ( { model | change = Unchanged }, Cmd.none )

                Unchanged ->
                    ( { model | change = Waiting }, getEl model.text )



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions _ =
    Time.every 500 Tick



-- VIEW


view model =
    Element.layout []
        (body model)


body : Model -> Element Msg
body model =
    row [ width fill, spacing 30 ]
        [ textInput model.text
        , resultView model
        ]


textInput : String -> Element Msg
textInput text =
    el
        [ width fill
        , Border.rounded 3
        , padding 30
        ]
        (Input.multiline
            [ height (px 600) ]
            { label = Input.labelHidden ""
            , onChange = EditedText
            , placeholder = Nothing
            , spellcheck = False
            , text = text
            }
        )


resultView : Model -> Element msg
resultView model =
    el [ width fill ] (text model.result)



-- HTTP


localApi : String
localApi =
    Url.absolute [ "el" ] []


getEl : String -> Cmd Msg
getEl text =
    Http.send NewEl (Http.post localApi (Http.jsonBody (Encode.string text)) elDecoder)


elDecoder =
    string


getCoreNLP : String -> Cmd Msg
getCoreNLP lang =
    let
        url =
            vildeApi lang "corenlp_3.8.0"

        postData =
            Http.jsonBody (Encode.string "This is a test.")
    in
        Http.send NewDocforia (Http.post url postData docforiaDecoder)


vildeApi : String -> String -> String
vildeApi lang config =
    Url.crossOrigin "http://vilde.cs.lth.se:9000" [ lang, config, "api", "json" ] []


docforiaDecoder =
    Decode.field "DM10" docforiaHelper


docforiaHelper =
    Decode.succeed Docforia
        |> required "text" string
        |> required "properties" propertiesDecoder
        |> required "edges" (list edgesDecoder)
        |> required "nodes" (list nodesDecoder)


propertiesDecoder =
    dict string


edgesDecoder =
    Decode.succeed Edges
        |> required "variants" (list string)
        |> required "layer" string
        |> required "properties" (list propertiesDecoder)
        |> required "connections" (list int)


nodesDecoder =
    Decode.succeed Nodes
        |> required "variants" (list string)
        |> required "layer" string
        |> required "properties" (list propertiesDecoder)
        |> required "ranges" (list int)
